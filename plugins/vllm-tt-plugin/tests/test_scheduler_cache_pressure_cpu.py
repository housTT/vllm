# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for TT prefill retry progress and fallback ownership.

Run without importing vLLM, torch or TT:
python plugins/vllm-tt-plugin/tests/test_scheduler_cache_pressure_cpu.py -v
The fixture executes production AST methods and real queue implementations.
"""

import collections
import contextlib
import sys
import typing
import unittest
from concurrent.futures import Future
from types import SimpleNamespace as N

from scheduler_cpu_fixture import (
    Mode,
    Policy,
    Status,
    build,
    dp_core,
    drain,
    extract,
    finish,
    g,
    lane_coordinator,
    req,
    step,
)


class CachePressureTests(unittest.TestCase):
    def test_original_resident_controls_sync_and_async(self):
        for asynchronous in (False, True):
            for residents in (25, 26, 27):
                with self.subTest(async_scheduling=asynchronous, residents=residents):
                    s, w, pool = build(residents)
                    s.scheduler_config.async_scheduling = asynchronous
                    outputs = [step(s) for _ in range(8)]
                    counts = [o.num_scheduled_tokens.get("waiting", 0) for o in outputs]
                    if residents == 25:
                        self.assertEqual(counts, [2048] * 4 + [1] * 4)
                        self.assertEqual(w.num_preemptions, 0)
                    elif residents == 26:
                        self.assertEqual(counts, [2048] + [0] * 7)
                        self.assertEqual(w.num_preemptions, 1)
                        self.assertEqual(outputs[1].preempted_req_ids, {"waiting"})
                        self.assertTrue(
                            all(len(o.num_scheduled_tokens) == 26 for o in outputs[1:])
                        )
                        self.assertEqual(pool.get_num_free_blocks(), 227)
                    else:
                        self.assertEqual(counts, [0] * 8)
                        self.assertEqual(w.num_preemptions, 0)
                    self.assertEqual(s.max_model_len, 262144)
                    self.assertEqual(s.max_num_running_reqs, 32)

    def test_finished_and_encoder_notifications_delivered_exactly_once(self):
        s, w, _ = build(26)
        step(s)
        s.finished_req_ids = {"old-finished"}
        freed = ["old-mm"]
        s.encoder_cache_manager.get_freed_mm_hashes = lambda: (
            [freed.pop()] if freed else []
        )
        out = step(s)
        self.assertEqual(out.finished_req_ids, {"old-finished"})
        self.assertEqual(out.preempted_req_ids, {"waiting"})
        self.assertEqual(out.free_encoder_mm_hashes, ["old-mm"])
        out = step(s)
        self.assertFalse(out.finished_req_ids)
        self.assertFalse(out.preempted_req_ids)
        self.assertFalse(out.free_encoder_mm_hashes)

    def test_completion_abort_and_no_decoder_release_retry(self):
        for removed in (["d0"], ["d0", "d1"], [f"d{i}" for i in range(26)]):
            with self.subTest(removed=len(removed)):
                s, w, _ = build(26)
                step(s)
                step(s)
                for rid in removed:
                    finish(s, rid)
                outputs = [step(s) for _ in range(4)]
                self.assertEqual(w.num_computed_tokens, 8192)
                self.assertEqual(w.num_preemptions, 1)
                self.assertEqual(outputs[0].finished_req_ids, set(removed))
                self.assertFalse(hasattr(w, "_tt_prefill_retry"))

    def test_aborted_waiter_does_not_block_next_request_or_id_reuse(self):
        s, w, _ = build(26)
        step(s)
        step(s)
        finish(s, "waiting")
        replacement = req("waiting", 0, 32, Status.WAITING)
        replacement.num_prompt_tokens = 32
        s.requests["waiting"] = replacement
        s.waiting.add_request(replacement)
        out = step(s)
        self.assertEqual(out.num_scheduled_tokens, {"waiting": 32})
        self.assertEqual(out.finished_req_ids, {"waiting"})
        self.assertFalse(hasattr(replacement, "_tt_prefill_retry"))

    def test_lower_priority_short_request_runs_without_reordering_waiters(self):
        for policy in (Policy.FCFS, Policy.PRIORITY):
            with self.subTest(policy=policy):
                s, w, _ = build(26)
                step(s)
                step(s)
                s.policy = policy
                s.waiting = g["create_request_queue"](policy)
                s.waiting.add_request(w)
                for index, size in enumerate((9000, 32, 9000)):
                    r = req(f"later{index}", 0, size, Status.WAITING)
                    r.num_prompt_tokens = size
                    r.priority = index + 1
                    r.arrival_time = index + 1
                    if size == 9000:
                        # A second previously failed continuation may coexist.
                        r._tt_prefill_retry = w._tt_prefill_retry
                    s.requests[r.request_id] = r
                    s.waiting.add_request(r)
                out = step(s)
                self.assertEqual(out.num_scheduled_tokens, {"later1": 32})
                self.assertEqual(
                    [r.request_id for r in s.waiting], ["waiting", "later0", "later2"]
                )
                self.assertEqual(w.num_preemptions, 1)

    def test_decode_page_boundary_capacity_change_allows_retry(self):
        s, w, pool = build(26)
        step(s)
        step(s)
        # At sliding eviction boundaries free capacity can grow without a
        # completion. Retry is allowed only when the recorded capacity improves.
        initial = pool.get_num_free_blocks()
        retried = False
        for _ in range(130):
            before = pool.get_num_free_blocks()
            out = step(s)
            if out.num_scheduled_tokens.get("waiting", 0) > 1:
                self.assertGreater(before, initial)
                retried = True
                break
        self.assertTrue(retried)
        for _ in range(3):
            step(s)
        self.assertEqual(w.num_computed_tokens, 8192)
        self.assertEqual(w.num_preemptions, 1)

    def test_pending_async_outputs_are_dropped_after_preemption_and_retry(self):
        s, w, _ = build(26)
        step(s)
        pending = g["_PendingOutputs"].for_request(w)
        pending.record()
        pending.record()
        w.num_output_placeholders = 2
        step(s)
        self.assertEqual((pending.outstanding, pending.stale), (2, 2))
        self.assertEqual(w.num_output_placeholders, 0)
        # Retry can start before either invalidated token returns.
        finish(s, "d0")
        step(s)
        computed = w.num_computed_tokens
        for token in (101, 102):
            self.assertEqual(s._update_request_with_output(w, [token]), ([], False))
            self.assertEqual(w.num_computed_tokens, computed)
            self.assertEqual(w.num_output_placeholders, 0)
        self.assertEqual((pending.outstanding, pending.stale), (0, 0))
        for _ in range(3):
            step(s)
        self.assertEqual(w.num_computed_tokens, 8192)
        self.assertEqual(w.num_preemptions, 1)

    def test_decode_preemption_does_not_create_continuation_guard(self):
        s, w, _ = build(26)
        r = s.running.pop()
        s._preempt_request(r, 0.0)
        self.assertFalse(hasattr(r, "_tt_prefill_retry"))

    def test_normal_short_prefill_and_decode_are_unchanged(self):
        s, w, _ = build(0)
        w.num_tokens = w.num_prompt_tokens = 32
        self.assertEqual(step(s).num_scheduled_tokens, {"waiting": 32})
        for _ in range(8):
            self.assertEqual(step(s).num_scheduled_tokens, {"waiting": 1})
        self.assertEqual(w.num_preemptions, 0)
        self.assertFalse(hasattr(w, "_tt_prefill_retry"))

    def test_forced_modes_keep_batches_pure_and_do_not_bypass_guard(self):
        s, w, _ = build(26)
        s.set_forced_mode(Mode.DECODE_ONLY)
        self.assertNotIn("waiting", step(s).num_scheduled_tokens)
        s.set_forced_mode(Mode.PREFILL_ONLY)
        self.assertEqual(step(s).num_scheduled_tokens, {"waiting": 2048})
        empty = step(s)
        self.assertEqual(empty.total_num_scheduled_tokens, 0)
        self.assertEqual(empty.preempted_req_ids, {"waiting"})
        for _ in range(3):
            self.assertEqual(step(s).total_num_scheduled_tokens, 0)
        s.set_forced_mode(Mode.DECODE_ONLY)
        self.assertEqual(len(step(s).num_scheduled_tokens), 26)
        self.assertEqual(w.num_preemptions, 1)
        s.set_forced_mode(Mode.DEFAULT)
        self.assertEqual(len(step(s).num_scheduled_tokens), 26)

    def test_gathered_dp_fallback_preserves_notifications_and_decodes(self):
        s, w, _ = build(26)
        core = dp_core(s)
        outputs = []
        for i in range(8):
            mode = core._dp_negotiate_forced_mode()
            core._dp_apply_forced_mode(mode)
            first = s.schedule()
            out = core._dp_schedule_with_zero_prefill_fallback(mode, first)
            outputs.append(out)
            drain(s, out)
        self.assertEqual(outputs[1].preempted_req_ids, {"waiting"})
        self.assertEqual(w.num_preemptions, 1)
        self.assertTrue(all(len(o.num_scheduled_tokens) == 26 for o in outputs[1:]))

    def test_lane_dp_fallback_preserves_each_lanes_notifications(self):
        s, w, _ = build(26)
        idle, unused, _ = build(0)
        idle.waiting.clear()
        lane = lane_coordinator([s, idle])
        first = lane.schedule()
        drain(s, first)
        idle.finished_req_ids = {"idle-finished"}
        out = lane.schedule()
        drain(s, out)
        self.assertEqual(out.preempted_req_ids, {"waiting"})
        self.assertEqual(out.finished_req_ids, {"idle-finished"})
        state = g["_get_tt_step_state"](out)
        self.assertEqual(state.lane_outputs[0].preempted_req_ids, {"waiting"})
        self.assertEqual(state.lane_outputs[1].finished_req_ids, {"idle-finished"})
        for _ in range(5):
            out = lane.schedule()
            drain(s, out)
            self.assertEqual(len(out.num_scheduled_tokens), 26)
        self.assertEqual(w.num_preemptions, 1)

    def test_connector_metadata_has_one_owner_and_decode_is_deferred(self):
        for connector in ("connector", "ec_connector"):
            for dp in (False, True):
                with self.subTest(connector=connector, dp=dp):
                    s, w, _ = build(26)
                    step(s)
                    metadata = []

                    def build_meta(output, metadata=metadata):
                        meta = object()
                        metadata.append(meta)
                        return meta

                    setattr(s, connector, N(build_connector_meta=build_meta))
                    core = dp_core(s) if dp else None
                    if dp:
                        mode = core._dp_negotiate_forced_mode()
                        core._dp_apply_forced_mode(mode)
                    out = s.schedule()
                    if dp:
                        out = core._dp_schedule_with_zero_prefill_fallback(mode, out)
                    field = (
                        "kv_connector_metadata"
                        if connector == "connector"
                        else "ec_connector_metadata"
                    )
                    self.assertEqual(out.total_num_scheduled_tokens, 0)
                    self.assertEqual(out.preempted_req_ids, {"waiting"})
                    self.assertIs(getattr(out, field), metadata[0])
                    self.assertEqual(len(metadata), 1)
                    if dp:
                        mode = core._dp_negotiate_forced_mode()
                        self.assertEqual(mode, Mode.DECODE_ONLY)
                        core._dp_apply_forced_mode(mode)
                    out = s.schedule()
                    self.assertEqual(len(out.num_scheduled_tokens), 26)
                    self.assertIs(getattr(out, field), metadata[1])
                    self.assertEqual(len(metadata), 2)

    def test_empty_output_reaches_runner_cleanup_before_early_return(self):
        Runner = extract(
            "plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py",
            "TTModelRunner",
            ["build_model_input"],
            bases=[],
        )
        runner = object.__new__(Runner)
        seen = []
        runner._update_states = seen.append
        out = g["SchedulerOutput"].make_empty()
        out.finished_req_ids = {"done"}
        out.preempted_req_ids = {"retry"}
        self.assertIsNone(runner.build_model_input(out, None))
        self.assertEqual(seen, [out])

    def test_both_gathered_dp_engine_steps_deliver_fallback_output(self):
        for asynchronous in (False, True):
            with self.subTest(async_scheduling=asynchronous):
                s, w, _ = build(26)
                core = dp_core(s)
                s.has_requests = lambda: True
                s.get_grammar_bitmask = lambda out: None
                s.update_from_output = lambda out, model: {}
                core._scheduler_paused = False
                core._dp_any_rank_has_scheduler_requests = lambda: True
                core._process_aborts_queue = lambda: None
                seen = []
                if asynchronous:
                    core.batch_queue = object()
                    core._dp_in_flight = None
                    core.is_ec_producer = False
                    core._dp_can_attempt_steady_decode_from_scheduler = (
                        lambda *args: False
                    )

                    def submit(out, grammar, seen=seen, **kw):
                        seen.append(out)
                        return N(scheduler_output=out)

                    core.dp_gather_submit = submit
                    core.dp_gather_finalize = lambda handle: object()
                    engine_step = core.step_dp_with_batch_queue
                else:

                    def execute(out, grammar, seen=seen):
                        seen.append(out)
                        return object()

                    core._execute_model_dp_gather = execute
                    engine_step = core.step
                for _ in range(3):
                    engine_step()
                    drain(s, seen[-1])
                self.assertEqual(seen[1].preempted_req_ids, {"waiting"})
                self.assertEqual(len(seen[2].num_scheduled_tokens), 26)
                self.assertEqual(w.num_preemptions, 1)

    def test_both_upstream_engine_steps_deliver_empty_output(self):
        g.update(cast=typing.cast, Future=Future, ModelRunnerOutput=object)
        Engine = extract(
            "vllm/v1/engine/core.py",
            "EngineCore",
            ["step", "step_with_batch_queue"],
            bases=[],
        )
        for asynchronous in (False, True):
            with self.subTest(async_scheduling=asynchronous):
                core = object.__new__(Engine)
                out = g["SchedulerOutput"].make_empty()
                out.preempted_req_ids = {"retry"}
                executed, updated = [], []
                future = Future()
                future.set_result(object())

                def execute(output, executed=executed, future=future, **kw):
                    executed.append(output)
                    return future

                def update(output, model, updated=updated):
                    updated.append(output)
                    return {}

                core.scheduler = N(
                    has_requests=lambda: True,
                    schedule=lambda out=out: out,
                    get_grammar_bitmask=lambda out: None,
                    update_from_output=update,
                )
                core.model_executor = N(execute_model=execute)
                core._scheduler_paused = False
                core._process_aborts_queue = lambda: None
                core.log_error_detail = lambda out: contextlib.nullcontext()
                core.log_iteration_details = lambda out: contextlib.nullcontext()
                core.batch_queue = collections.deque()
                core.batch_queue_size = 2
                core.is_ec_producer = core.is_pooling_model = False
                result = core.step_with_batch_queue() if asynchronous else core.step()
                self.assertEqual(result, ({}, False))
                self.assertEqual(executed, [out])
                self.assertEqual(updated, [out])

    def test_finite_cache_pressure_workload_drains_all_requests(self):
        s, w, pool = build(26)
        # A bounded CPU liveness control with 31 outstanding / 62 total requests.
        # The seeded resident requests have already generated three tokens;
        # this is not a device benchmark or a measurement of the release row.
        launched = 27

        def add_waiter():
            nonlocal launched
            r = req(f"new{launched}", 0, 8192, Status.WAITING)
            s.requests[r.request_id] = r
            s.waiting.add_request(r)
            launched += 1

        for _ in range(4):
            add_waiter()
        completed = 0
        for _ in range(3000):
            step(s)
            for rid, r in list(s.requests.items()):
                if r.num_tokens >= r.num_prompt_tokens + r.max_tokens:
                    finish(s, rid)
                    completed += 1
                    if launched < 62:
                        add_waiter()
            self.assertLessEqual(len(s.requests), 31)
            if not s.requests:
                break
        self.assertEqual(completed, 62)
        self.assertEqual(pool.get_num_free_blocks(), 4127)

    def test_no_device_imports(self):
        self.assertFalse(
            any(
                m == "torch" or m == "ttnn" or m.startswith("vllm") for m in sys.modules
            )
        )


if __name__ == "__main__":
    unittest.main()
