# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU controls coupling production TT history updates, scheduler and sampling.

Tensor storage and random draws are real CPU torch/numpy. TT modules are never
imported; block-table/logits-processor collaborators are inert substitutes.
"""

import ast
import copy
import dataclasses
import sys
import types
import typing
import unittest
from types import SimpleNamespace as N

import test_recompute_tail as tail
from scheduler_cpu_fixture import Status, build
from scheduler_cpu_fixture import g as scheduler_globals

np, torch = tail.np, tail.torch
G = dict(tail.GLOBALS)
G.update(
    SamplingMetadata=N,
    cast=typing.cast,
    dataclass=dataclasses.dataclass,
    copy=copy,
    __name__=__name__,
    length_from_prompt_token_ids_or_embeds=lambda ids, embeds: len(ids),
    SEED_NONE_SENTINEL=-1,
    LOGPROBS_NONE_SENTINEL=-2,
)


def execute_node(node, path):
    module = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], []
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), G)


state_path = tail.ROOT / "vllm/v1/worker/gpu_input_batch.py"
execute_node(
    next(
        n
        for n in ast.parse(state_path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "CachedRequestState"
    ),
    state_path,
)
for path in (
    tail.PLUGIN / "input_batch.py",
    tail.ROOT / "vllm/v1/sample/ops/topk_topp_sampler.py",
):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "apply_cached_req_state_update",
            "preempt_cached_request_state",
            "restore_request_rng",
            "acknowledge_request_outputs",
            "checkpoint_request_rng",
            "clone_torch_generator",
            "random_sample",
        }:
            execute_node(node, path)


class BlockTable:
    def __init__(self, **kw):
        self.rows = {}

    def add_row(self, blocks, row):
        self.rows[row] = copy.deepcopy(blocks)

    def append_row(self, blocks, row):
        self.rows[row] = tuple(old + new for old, new in zip(self.rows[row], blocks))

    def move_row(self, src, dst):
        self.rows[dst] = self.rows[src]


class BatchUpdates:
    def __init__(self):
        self.added = []

    def removed_append(self, row):
        pass


G.update(
    MultiGroupBlockTable=BlockTable,
    BatchUpdateBuilder=BatchUpdates,
    LogitsProcessors=lambda: N(all=lambda: []),
)
execute_node(
    next(
        n
        for n in ast.parse((tail.PLUGIN / "input_batch.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "SamplingInputBatch"
    ),
    tail.PLUGIN / "input_batch.py",
)
Batch = tail.extract(
    tail.PLUGIN / "input_batch.py",
    "InputBatch",
    {
        "__init__",
        "add_request",
        "remove_request",
        "advance_generators",
        "req_ids",
        "num_reqs",
        "all_greedy",
        "no_penalties",
        "no_allowed_token_ids",
        "max_num_logprobs",
        "condense",
    },
    G,
)
Lane = tail.extract(
    tail.PLUGIN / "input_batch.py",
    "TTLaneInputBatch",
    {
        "apply_step_plan",
        "build_merged_sampling_metadata",
    },
    G,
)
Runner = tail.extract(
    tail.PLUGIN / "model_runner.py",
    "TTModelRunner",
    {
        "_update_states",
        "_prepare_model_inputs",
        "_apply_sampled_tokens_to_state",
        "_build_host_generators",
        "_build_chunked_prefill_output",
    },
    G,
)


def worker_state(outputs=5, seed=None, rid="waiting", prompt=8192):
    sp = N(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=seed,
        logprobs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
    )
    return G["CachedRequestState"](
        req_id=rid,
        prompt_token_ids=list(range(prompt)),
        mm_features=[],
        sampling_params=sp,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        block_ids=([1],),
        num_computed_tokens=prompt + outputs - 1,
        output_token_ids=list(range(prompt, prompt + outputs)),
    )


def batch_for(*requests):
    batch = Batch(4, 262144, 2048, 128, [128], [128])
    batch.refresh_logitsprocs = lambda: None
    batch.block_tables_for_rows = lambda rows, width: [
        torch.zeros(len(rows), width, dtype=torch.int32)
    ]
    batch.reset_slot_remap = lambda: None
    for request in requests:
        batch.add_request(request)
    return batch


def runner_for(*requests):
    runner = tail.runner()
    runner.input_batch = batch_for(*requests)
    runner.requests = {r.req_id: r for r in requests}
    runner.model_config.max_model_len = 262144
    runner.tt_per_lane_max_num_seqs = 4
    runner._release_dead_state_slots = lambda out: None
    runner.encoder_cache = {}
    runner._apply_sampled_tokens_to_state = types.MethodType(
        Runner._apply_sampled_tokens_to_state, runner
    )
    return runner


def empty_output():
    return scheduler_globals["SchedulerOutput"].make_empty()


def real_resumed_scheduler(outputs=5, pending=1):
    s, r, _ = build(0)
    s.use_pp = False

    def make_cached(*args):
        data = tail.Scheduler._make_cached_request_data(s, *args)
        data.num_reqs = len(data.req_ids)
        return data

    s._make_cached_request_data = make_cached
    r.num_tokens = 8192 + outputs
    r.all_token_ids = list(range(r.num_tokens))
    r.num_computed_tokens = r.num_tokens - 1
    r.num_output_placeholders = pending
    r.status = Status.RUNNING
    r.is_prefill_chunk = False
    s.waiting.clear()
    for _ in range(pending):
        scheduler_globals["_PendingOutputs"].for_request(r).record()
    s._preempt_request(r, 0.0)
    notification = empty_output()
    notification.preempted_req_ids = {r.request_id}
    s._finalize_scheduler_output(notification)
    return s, r, notification


def draw(batch):
    generators = Runner._build_host_generators(batch, [0], None)
    return int(G["random_sample"](torch.full((1, 128), 1 / 128), generators)[0])


class AcceptedHistoryTest(unittest.TestCase):
    def test_stale_history_before_or_after_resume_and_full_tail_replay(self):
        for pending in (1, 2):
            for arrival in ("before_preempt", "before_resume", "after_resume"):
                for tail_chunks in ((5,), (4, 1)):
                    with self.subTest(
                        pending=pending, arrival=arrival, tail=tail_chunks
                    ):
                        old = worker_state()
                        runner = runner_for(old)
                        s, r, notification = real_resumed_scheduler(pending=pending)

                        def stale(pending=pending, runner=runner, old=old, s=s, r=r):
                            for i in range(pending):
                                runner._apply_sampled_tokens_to_state(
                                    torch.tensor([[8197 + i]]),
                                    req_ids=["waiting"],
                                    request_states=(old,),
                                )
                                self.assertEqual(
                                    s._update_request_with_output(r, [8197 + i]),
                                    ([], False),
                                )

                        if arrival == "before_preempt":
                            stale()
                        Runner._update_states(runner, notification)
                        self.assertIsNot(runner.requests["waiting"], old)
                        if arrival == "before_resume":
                            stale()
                        outputs = []
                        for i in range(4 + len(tail_chunks)):
                            if i >= 4:
                                s.max_num_scheduled_tokens = tail_chunks[i - 4]
                            out = s.schedule()
                            Runner._update_states(runner, out)
                            if arrival == "after_resume" and i == 0:
                                stale()
                            current = runner.requests["waiting"]
                            self.assertEqual(len(current.output_token_ids), 5)
                            row = runner.input_batch.req_id_to_index["waiting"]
                            self.assertEqual(runner.input_batch.num_tokens[row], 8197)
                            self.assertEqual(
                                runner.input_batch.req_output_token_ids[row],
                                list(range(8192, 8197)),
                            )
                            self.assertEqual(
                                runner.input_batch.block_table.rows[row],
                                current.block_ids,
                            )
                            mi = Runner._prepare_model_inputs(runner, out, None)
                            intermediate = i != 3 + len(tail_chunks)
                            self.assertEqual(
                                mi.intermediate_prefill_mask.tolist(), [intermediate]
                            )
                            start = int(mi.input_positions[0])
                            end = int(mi.prompt_lens[0])
                            self.assertEqual(
                                mi.input_tokens[0, start:end].tolist(),
                                list(range(start, end)),
                            )
                            outputs.extend(range(start, end))
                        self.assertEqual(outputs, list(range(8197)))
                        result = Runner._build_chunked_prefill_output(
                            runner,
                            ["waiting"],
                            torch.tensor([[8197]]),
                            None,
                            np.array([False]),
                        )
                        self.assertEqual(result.sampled_token_ids, [[8197]])
                        self.assertEqual(len(current.output_token_ids), 6)
                        # Next normal decode may share a batch with a resident.
                        current.request_id = current.req_id
                        current.all_token_ids = (
                            current.prompt_token_ids + current.output_token_ids
                        )
                        current.num_output_tokens = 6
                        current.num_computed_tokens = 8197
                        other = tail.request(128, 5, computed=132, req_id="resident")
                        r.num_tokens = 8198
                        r.num_output_placeholders = 0
                        r.all_token_ids.append(8197)
                        mi, _ = tail.prepare(
                            runner,
                            [current, other],
                            [1, 1],
                            scheduler_requests=[r, other],
                        )
                        self.assertIsNone(mi.prompt_lens)

    def test_seeded_rng_rewinds_complete_draw_sequence(self):
        for pending in (1, 2):
            for completion_after_preempt in (False, True):
                with self.subTest(pending=pending, late=completion_after_preempt):
                    control = worker_state(outputs=0, seed=10)
                    request = worker_state(outputs=0, seed=10)
                    cb, rb = batch_for(control), batch_for(request)
                    for i in range(5):
                        expected, actual = draw(cb), draw(rb)
                        self.assertEqual(actual, expected)
                        control.output_token_ids.append(expected)
                        request.output_token_ids.append(actual)
                        G["acknowledge_request_outputs"](request, i + 1)
                        self.assertEqual(request._tt_rng_checkpoints, {})
                    expected = draw(cb)
                    for _ in range(pending):
                        request.output_token_ids.append(draw(rb))
                    resumed = G["preempt_cached_request_state"](request, 5)
                    if completion_after_preempt:
                        # A late sampler still owns the old generator object.
                        G["random_sample"](
                            torch.full((1, 128), 1 / 128), {0: request.generator}
                        )
                    G["apply_cached_req_state_update"](resumed, 0, ([2],), True, 5)
                    actual = draw(batch_for(resumed))
                    self.assertEqual((expected, actual), (108, 108))
                    self.assertEqual(len(resumed.output_token_ids), 5)

    def test_resume_without_prior_notification_reconciles_history_and_rng(self):
        request = worker_state(outputs=0, seed=10)
        runner = runner_for(request)
        for _ in range(5):
            request.output_token_ids.append(draw(runner.input_batch))
        expected = draw(runner.input_batch)
        request.output_token_ids.append(expected)
        s, req, notification = real_resumed_scheduler()
        req.all_token_ids = req.all_token_ids[:8192] + request.output_token_ids[:5]
        out = s.schedule()
        Runner._update_states(runner, out)
        resumed = runner.requests["waiting"]
        self.assertIsNot(resumed, request)
        self.assertEqual(draw(runner.input_batch), expected)
        before = list(resumed.output_token_ids)
        runner._apply_sampled_tokens_to_state(
            torch.tensor([[123]]), req_ids=["waiting"], request_states=(request,)
        )
        self.assertEqual(resumed.output_token_ids, before)

    def test_normal_async_update_and_pause_keep_history_identity_and_rng(self):
        request = worker_state(outputs=5, seed=10)
        runner = runner_for(request)
        before = request.generator.get_state().clone()
        G["apply_cached_req_state_update"](request, 8198, ([2],), False, 7)
        self.assertEqual(len(request.output_token_ids), 5)
        self.assertTrue(torch.equal(request.generator.get_state(), before))
        Runner._update_states(runner, empty_output())
        self.assertIs(runner.requests["waiting"], request)
        runner._apply_sampled_tokens_to_state(
            torch.tensor([[42]]), req_ids=["waiting"], request_states=(request,)
        )
        self.assertEqual(request.output_token_ids[-1], 42)

    def test_accepted_annotation_ignores_placeholders_and_survives_lane_merge(self):
        s, r, preempt = real_resumed_scheduler(pending=2)
        self.assertEqual(preempt._tt_accepted_output_tokens, {"waiting": 5})
        out = s.schedule()
        self.assertEqual(out._tt_accepted_output_tokens, {"waiting": 5})
        merged = scheduler_globals["merge_lane_scheduler_outputs"]([out])
        self.assertEqual(merged._tt_accepted_output_tokens, {"waiting": 5})
        replacement = empty_output()
        scheduler_globals["carry_scheduler_notifications"](preempt, replacement)
        self.assertEqual(replacement._tt_accepted_output_tokens, {"waiting": 5})

    def test_checkpoint_memory_tracks_only_unacknowledged_outputs(self):
        request = worker_state(outputs=0, seed=10)
        batch = batch_for(request)
        for i in range(100):
            token = draw(batch)
            request.output_token_ids.append(token)
            # Two in-flight draws are retained, with no fixed-size truncation.
            G["acknowledge_request_outputs"](request, max(0, i - 1))
            self.assertLessEqual(len(request._tt_rng_checkpoints), 2)
        self.assertEqual(sorted(request._tt_rng_checkpoints), [98, 99])

    def test_front_and_gathered_rng_sequence_and_checkpoint_ordinal(self):
        for gathered in (False, True):
            with self.subTest(gathered=gathered):
                request = worker_state(outputs=0, seed=10)
                batch = batch_for(request)
                baseline = torch.Generator().manual_seed(10)
                for i in range(12):
                    # The historical gathered path samples serialized copies;
                    # the local canonical generator only consumes rand(1).
                    torch.rand(1, generator=baseline)
                    baseline_draw = copy.deepcopy(baseline) if gathered else baseline
                    expected = int(
                        G["random_sample"](
                            torch.full((1, 128), 1 / 128), {0: baseline_draw}
                        )[0]
                    )
                    generators = Runner._build_host_generators(batch, [0], None)
                    self.assertEqual(sorted(request._tt_rng_checkpoints), [i])
                    self.assertEqual(request._tt_next_rng_output, i + 1)
                    if gathered:
                        generators = copy.deepcopy(generators)
                    actual = int(
                        G["random_sample"](torch.full((1, 128), 1 / 128), generators)[0]
                    )
                    self.assertEqual(actual, expected)
                    request.output_token_ids.append(actual)
                    G["acknowledge_request_outputs"](request, i + 1)
                    self.assertTrue(
                        torch.equal(request.generator.get_state(), baseline.get_state())
                    )
                # Intermediate prefill gets a clone and adds no checkpoint.
                before = request.generator.get_state().clone()
                clone = Runner._build_host_generators(batch, [0], torch.tensor([True]))
                G["random_sample"](torch.full((1, 128), 1 / 128), clone)
                self.assertEqual(request._tt_rng_checkpoints, {})
                self.assertEqual(request._tt_next_rng_output, 12)
                self.assertTrue(torch.equal(request.generator.get_state(), before))
                # Resume also preserves the gathered copy semantics: only the
                # canonical rand advancement is rolled back in that path.
                torch.rand(1, generator=baseline)
                baseline_draw = copy.deepcopy(baseline) if gathered else baseline
                expected = int(
                    G["random_sample"](
                        torch.full((1, 128), 1 / 128), {0: baseline_draw}
                    )[0]
                )
                generators = Runner._build_host_generators(batch, [0], None)
                if gathered:
                    generators = copy.deepcopy(generators)
                request.output_token_ids.append(
                    int(
                        G["random_sample"](torch.full((1, 128), 1 / 128), generators)[0]
                    )
                )
                resumed = G["preempt_cached_request_state"](request, 12)
                generators = Runner._build_host_generators(
                    batch_for(resumed), [0], None
                )
                if gathered:
                    generators = copy.deepcopy(generators)
                actual = int(
                    G["random_sample"](torch.full((1, 128), 1 / 128), generators)[0]
                )
                self.assertEqual(actual, expected)

    def test_lane_preempt_resume_and_sampling_checkpoint_ownership(self):
        request = worker_state(outputs=0, seed=10)
        other = worker_state(outputs=0, seed=11, rid="other")
        batch = batch_for(request, other)
        requests = {"waiting": request, "other": other}
        batch.add_request_to_row = batch.add_request
        plan = N(req_id_to_row={"waiting": 0, "other": 1})
        baseline = torch.Generator().manual_seed(10)
        other_before = other.generator.get_state().clone()
        for i in range(5):
            metadata = Lane.build_merged_sampling_metadata(batch, scheduled_rows=[0])
            self.assertEqual(sorted(request._tt_rng_checkpoints), [i])
            self.assertEqual(request._tt_next_rng_output, i + 1)
            actual = int(
                G["random_sample"](torch.full((1, 128), 1 / 128), metadata.generators)[
                    0
                ]
            )
            expected = int(
                G["random_sample"](torch.full((1, 128), 1 / 128), {0: baseline})[0]
            )
            self.assertEqual(actual, expected)
            request.output_token_ids.append(actual)
            G["acknowledge_request_outputs"](request, i + 1)
        expected = int(
            G["random_sample"](torch.full((1, 128), 1 / 128), {0: baseline})[0]
        )
        intermediate = Lane.build_merged_sampling_metadata(
            batch, scheduled_rows=[0], non_sampling_rows=[0]
        )
        G["random_sample"](torch.full((1, 128), 1 / 128), intermediate.generators)
        self.assertEqual(request._tt_rng_checkpoints, {})
        metadata = Lane.build_merged_sampling_metadata(batch, scheduled_rows=[0])
        request.output_token_ids.append(
            int(
                G["random_sample"](torch.full((1, 128), 1 / 128), metadata.generators)[
                    0
                ]
            )
        )
        s, req, notification = real_resumed_scheduler()
        req.all_token_ids = req.all_token_ids[:8192] + request.output_token_ids[:5]
        Lane.apply_step_plan(batch, notification, plan, requests, {})
        self.assertIsNot(requests["waiting"], request)
        self.assertIs(requests["other"], other)
        out = s.schedule()
        Lane.apply_step_plan(batch, out, plan, requests, {})
        resumed = requests["waiting"]
        self.assertEqual(len(resumed.output_token_ids), 5)
        self.assertEqual(batch.num_tokens[0], 8197)
        self.assertEqual(batch.req_output_token_ids[0], resumed.output_token_ids)
        metadata = Lane.build_merged_sampling_metadata(batch, scheduled_rows=[0])
        actual = int(
            G["random_sample"](torch.full((1, 128), 1 / 128), metadata.generators)[0]
        )
        self.assertEqual(actual, expected)
        self.assertEqual(sorted(resumed._tt_rng_checkpoints), [5])
        self.assertTrue(torch.equal(other.generator.get_state(), other_before))

    def test_no_tt_or_vllm_imports(self):
        self.assertFalse(
            any(
                n == "ttnn" or n.startswith(("vllm", "vllm_tt_plugin", "ttnn."))
                for n in sys.modules
            )
        )


if __name__ == "__main__":
    unittest.main()
