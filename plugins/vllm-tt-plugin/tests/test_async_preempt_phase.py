# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actual-source scheduler metadata tests; no runtime/model/tensor imports.

The harness AST-loads complete bookkeeping methods and the runner's phase
selection/guard. Cache allocation and execution are metadata stand-ins. Synthetic
output IDs only exercise the scheduler contract; no model outputs are read.
"""

import ast
import os
import pickle
import sys
import unittest
from collections import deque
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

# The override permits reviewing an unapplied test outside the checkout. It is
# not needed when this file lives in plugins/vllm-tt-plugin/tests/.
VLLM_ROOT = Path(
    os.environ.get("VLLM_TT_TEST_ROOT", Path(__file__).resolve().parents[3])
).resolve()
LIVE = VLLM_ROOT / "plugins/vllm-tt-plugin/src/vllm_tt_plugin"
UPSTREAM = VLLM_ROOT / "vllm/v1/core/sched"
SOURCE = LIVE


def definition(path, name, parent=None):
    nodes = ast.parse(path.read_text()).body
    if parent:
        nodes = next(
            n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent
        )
    return next(
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )


def execute(nodes, namespace, filename):
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), namespace)


class Status(Enum):
    RUNNING = 1
    PREEMPTED = 2


class Queue(deque):
    def prepend_request(self, request):
        self.appendleft(request)

    def prepend_requests(self, requests):
        self.extendleft(reversed(requests))


class Request:
    def __init__(self, prompt=1536, outputs=9000):
        self.request_id = "metadata-request"
        self.num_prompt_tokens = prompt
        self.output_count = outputs
        self.num_computed_tokens = self.num_tokens - 1
        self.num_output_placeholders = 0
        self.is_prefill_chunk = False
        self.status = Status.RUNNING
        self.spec_token_ids = []
        self.num_preemptions = 0
        self.discard_latest_async_tokens = False
        self.use_structured_output = False
        self.has_encoder_inputs = False

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + self.output_count

    def append_output_token_ids(self, _synthetic_id):
        self.output_count += 1


def load_scheduler():
    module = ModuleType("metadata_scheduler_source")
    module.__dict__.update(
        dataclass=dataclass,
        Enum=Enum,
        ClassVar=ClassVar,
        RequestStatus=Status,
        SchedulerInterface=object,
        create_request_queue=lambda _: Queue(),
        check_stop=lambda *_: False,
    )
    sys.modules[module.__name__] = module
    base = definition(UPSTREAM / "scheduler.py", "Scheduler")
    keep = {"_preempt_request", "_update_after_schedule", "_update_request_with_output"}
    base.body = [
        n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in keep
    ]
    execute([base], module.__dict__, UPSTREAM / "scheduler.py")

    def metadata_schedule(self):
        output = self.next_output
        self._update_after_schedule(output)
        return output

    module.Scheduler.schedule = metadata_schedule
    execute(
        [definition(UPSTREAM / "async_scheduler.py", "AsyncScheduler")],
        module.__dict__,
        UPSTREAM / "async_scheduler.py",
    )
    execute(
        [
            definition(SOURCE / "scheduler.py", n)
            for n in ("_PendingOutputs", "TTSchedulingMode", "TTScheduler")
        ],
        module.__dict__,
        SOURCE / "scheduler.py",
    )
    return module


MODULE = load_scheduler()


def scheduler(async_mode=True):
    result = object.__new__(MODULE.TTScheduler)
    result.scheduler_config = SimpleNamespace(async_scheduling=async_mode)
    result.kv_cache_manager = SimpleNamespace(
        free=lambda _: None, cache_blocks=lambda *_: None
    )
    result.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    result._spec_token_placeholders = []
    result.max_model_len = 262144
    result.max_num_running_reqs = 32
    result.log_stats = False
    result.policy = "fcfs"
    result.running = []
    result.waiting = Queue()
    result.requests = {}
    return result


def output(requests, counts, resumed=False):
    return SimpleNamespace(
        num_scheduled_tokens={r.request_id: n for r, n in zip(requests, counts)},
        total_num_scheduled_tokens=sum(counts),
        scheduled_spec_decode_tokens={},
        pending_structured_output_tokens=False,
        has_structured_output_requests=False,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r.request_id for r in requests],
            num_reqs=len(requests),
            resumed_req_ids={r.request_id for r in requests} if resumed else set(),
        ),
    )


def phase_from_actual_source(step, before, histories, prompts):
    """Execute the exact phase predicates and mixed-batch assertion, no tensors."""
    fn = definition(
        SOURCE / "model_runner.py", "_prepare_model_inputs", "TTModelRunner"
    )
    assignments = {}
    for node in fn.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node
    names = (
        "cached_reqs",
        "num_sched",
        "prefill_req_ids",
        "has_chunked_continuation",
        "is_prompt",
    )
    body = [assignments[name] for name in names if name in assignments]
    branch = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "is_prompt"
    )
    body.append(ast.If(test=branch.test, body=[branch.body[0]], orelse=[]))
    batch = SimpleNamespace(
        req_id_to_index={r: i for i, r in enumerate(step.num_scheduled_tokens)},
        num_computed_tokens_cpu=before,
        num_prompt_tokens=prompts,
        num_tokens=histories,
    )
    namespace = {"input_batch": batch, "scheduler_output": step}
    execute(body, namespace, SOURCE / "model_runner.py")
    return namespace["is_prompt"]


def schedule_step(sched, req, count, prefill=False, resumed=False):
    before = req.num_computed_tokens
    history = req.num_tokens
    step = output([req], [count], resumed)
    sched.next_output = step
    sched.requests[req.request_id] = req
    sched.running = [req]
    step = sched._schedule_prefill_only() if prefill else sched._schedule_decode_only()
    phase = phase_from_actual_source(step, [before], [history], [req.num_prompt_tokens])
    return step, phase, before + count < history


def complete(sched, req, phase, intermediate):
    # Exact ModelRunner output contract: intermediate prefill -> [], else one
    # sample. Upstream update_from_output invokes this method only if nonempty.
    generated = [] if phase and intermediate else [1]
    if generated:
        return sched._update_request_with_output(req, generated)
    return [], False


class AsyncPreemptionMetadata(unittest.TestCase):
    def test_pending_outputs_discard_in_order_across_repeated_preemption(self):
        s = scheduler()
        r = Request()
        for _ in range(2):
            schedule_step(s, r, 1)
        pending = MODULE._PendingOutputs.for_request(r)
        self.assertEqual((pending.outstanding, r.num_output_placeholders), (2, 2))
        s._preempt_request(r, 0.0)
        self.assertEqual((pending.stale, r.num_output_placeholders), (2, 0))
        self.assertEqual(s._update_request_with_output(r, [1]), ([], False))
        r.status = Status.RUNNING
        s._preempt_request(r, 1.0)
        self.assertEqual(s._update_request_with_output(r, [1]), ([], False))
        self.assertEqual(
            (pending.outstanding, pending.stale, r.num_output_placeholders), (0, 0, 0)
        )
        self.assertEqual(r.output_count, 9000)

    def test_ordinary_decode_sync_and_async_remains_decode(self):
        for async_mode in (False, True):
            s = scheduler(async_mode)
            r = Request()
            for _ in range(3):
                _, phase, intermediate = schedule_step(s, r, 1)
                self.assertFalse(phase)
                self.assertEqual(complete(s, r, phase, intermediate), ([1], False))
                self.assertEqual(r.num_output_placeholders, 0)

    def test_initial_partial_and_final_prefill(self):
        s = scheduler()
        r = Request(prompt=4097, outputs=0)
        r.num_computed_tokens = 0
        r.is_prefill_chunk = True
        for count in (2048, 2048, 1):
            _, phase, intermediate = schedule_step(s, r, count, prefill=True)
            self.assertTrue(phase)
            emitted, _ = complete(s, r, phase, intermediate)
            self.assertEqual(len(emitted), int(not intermediate))
        self.assertEqual((r.output_count, r.num_output_placeholders), (1, 0))

    def test_recovery_across_original_prompt_preserves_zero_intermediate_output(self):
        for prompt, outputs in ((1536, 9000), (2048, 8192), (3072, 8193), (1536, 8705)):
            with self.subTest(prompt_length=prompt, committed_output_count=outputs):
                s = scheduler()
                r = Request(prompt, outputs)
                schedule_step(s, r, 1)
                s._preempt_request(r, 0.0)
                self.assertEqual(s._update_request_with_output(r, [1]), ([], False))
                r.status = Status.RUNNING
                r.is_prefill_chunk = True
                resumed = True
                target = r.num_tokens
                while r.num_computed_tokens < target:
                    count = min(2048, target - r.num_computed_tokens)
                    _, phase, intermediate = schedule_step(
                        s, r, count, prefill=True, resumed=resumed
                    )
                    resumed = False
                    # With the unmodified runner, the second/third recovery
                    # chunk returns a sample and this executes the actual
                    # AsyncScheduler assertion at placeholder 0 -> -1.
                    emitted, _ = complete(s, r, phase, intermediate)
                    self.assertTrue(phase)
                    self.assertEqual(len(emitted), int(not intermediate))
                    self.assertEqual(r.num_output_placeholders, 0)
                self.assertEqual(r.output_count, outputs + 1)
                self.assertEqual(MODULE._PendingOutputs.for_request(r).outstanding, 0)

    def test_mixed_one_token_final_and_partial_recovery(self):
        s = scheduler()
        a = Request(prompt=1536, outputs=8705)
        b = Request(prompt=1536, outputs=10000)
        b.request_id = "metadata-request-b"
        a.num_computed_tokens = 10240
        b.num_computed_tokens = 8192
        a.is_prefill_chunk = b.is_prefill_chunk = True
        step = output([a, b], [1, 2048])
        s.requests = {r.request_id: r for r in (a, b)}
        s.running = [a, b]
        s.next_output = step
        s._schedule_prefill_only()
        phase = phase_from_actual_source(
            step, [10240, 8192], [10241, 11536], [1536, 1536]
        )
        self.assertTrue(phase)
        self.assertEqual(complete(s, a, phase, False), ([1], False))
        self.assertEqual(complete(s, b, phase, True), ([], False))
        self.assertEqual((a.num_output_placeholders, b.num_output_placeholders), (0, 0))

    def test_marked_recovery_with_unmarked_decode_still_rejects(self):
        a = Request(prompt=1536, outputs=9000)
        b = Request(prompt=1536, outputs=9000)
        b.request_id = "metadata-request-b"
        step = output([a, b], [2048, 1])
        step._tt_prefill_req_ids = {a.request_id}
        with self.assertRaisesRegex(AssertionError, "Prefill batch should not include"):
            phase_from_actual_source(step, [2048, 10535], [10536, 10536], [1536, 1536])

    def test_all_intermediate_recovery_has_no_output_or_placeholders(self):
        s = scheduler()
        requests = [Request(), Request()]
        requests[1].request_id = "metadata-request-b"
        for r in requests:
            r.num_computed_tokens = 2048
            r.is_prefill_chunk = True
        step = output(requests, [2048, 2048])
        s.next_output = step
        s.running = requests
        s.requests = {r.request_id: r for r in requests}
        s._schedule_prefill_only()
        phase = phase_from_actual_source(
            step, [2048, 2048], [10536, 10536], [1536, 1536]
        )
        for r in requests:
            self.assertEqual(complete(s, r, phase, True), ([], False))
            self.assertEqual(
                (
                    r.num_output_placeholders,
                    MODULE._PendingOutputs.for_request(r).outstanding,
                ),
                (0, 0),
            )
            self.assertEqual(r.output_count, 9000)

    def test_prefill_marker_does_not_leak_into_decode_fallback(self):
        s = scheduler()
        r = Request()
        r.is_prefill_chunk = False
        s.running = [r]
        s.next_output = output([], [])
        prefill_output = s._schedule_prefill_only()
        self.assertFalse(getattr(prefill_output, "_tt_prefill_req_ids", set()))
        _, phase, intermediate = schedule_step(s, r, 1)
        self.assertFalse(phase)
        self.assertEqual(complete(s, r, phase, intermediate), ([1], False))

    def test_actual_output_dataclass_pickle_copy_and_uniproc_worker_forwarding(self):
        module = ModuleType("metadata_output_source")
        sys.modules[module.__name__] = module
        module.__dict__.update(
            dataclass=dataclass,
            cached_property=cached_property,
            bc_linter_include=lambda x: x,
        )
        execute(
            [
                definition(UPSTREAM / "output.py", name)
                for name in ("CachedRequestData", "SchedulerOutput")
            ],
            module.__dict__,
            UPSTREAM / "output.py",
        )
        step = module.SchedulerOutput.make_empty()
        step._tt_prefill_req_ids = {"metadata-request"}
        for cloned in (
            deepcopy(step),
            pickle.loads(pickle.dumps(step, protocol=4)),
            pickle.loads(pickle.dumps(step, protocol=5)),
        ):
            self.assertEqual(cloned._tt_prefill_req_ids, step._tt_prefill_req_ids)

        # The native uni executor directly passes the same object to TTWorker.
        ns = {
            "Future": Future,
            "Any": object,
            "AsyncModelRunnerOutput": type("UnusedAsyncOutput", (), {}),
            "run_method": lambda worker, method, args, kwargs: getattr(worker, method)(
                *args, **kwargs
            ),
        }
        path = VLLM_ROOT / "vllm/v1/executor/uniproc_executor.py"
        node = definition(path, "UniProcExecutor")
        node.bases = []
        node.body = [
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("execute_model", "collective_rpc")
        ]
        execute([node], ns, path)
        execute(
            [definition(LIVE / "worker.py", "execute_model", "TTWorker")],
            ns,
            LIVE / "worker.py",
        )
        worker = SimpleNamespace(
            is_driver_worker=True,
            model_runner=SimpleNamespace(execute_model=lambda received: received),
        )
        worker.execute_model = lambda received: ns["execute_model"](worker, received)
        executor = ns["UniProcExecutor"]()
        executor.driver_worker = worker
        self.assertIs(executor.execute_model(step), step)
        self.assertIs(executor.execute_model(step, non_block=True).result(), step)

    def test_lane_dispatch_uses_its_existing_phase_plan(self):
        namespace = {}
        path = LIVE / "input_batch.py"
        execute(
            [definition(path, "build_model_input", "TTLaneInputBatch")], namespace, path
        )
        lane = SimpleNamespace(
            _build_decode_input=lambda *_: "decode",
            _build_prefill_input=lambda *_: "prefill",
        )
        step = output([], [])  # Merged lane output need not have our new marker.
        for is_decode, expected in ((False, "prefill"), (True, "decode")):
            self.assertEqual(
                namespace["build_model_input"](
                    lane, None, step, None, SimpleNamespace(is_decode=is_decode)
                ),
                expected,
            )

    def test_unmarked_mixed_decode_prefill_guard_still_rejects(self):
        a = Request(prompt=4096, outputs=0)
        b = Request(prompt=1536, outputs=9000)
        b.request_id = "metadata-request-b"
        step = output([a, b], [2048, 1])
        with self.assertRaisesRegex(AssertionError, "Prefill batch should not include"):
            phase_from_actual_source(step, [0, 10535], [4096, 10536], [4096, 1536])


if __name__ == "__main__":
    unittest.main(verbosity=2)
