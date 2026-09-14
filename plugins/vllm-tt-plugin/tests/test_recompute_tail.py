# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for resumed prefill crossing the original prompt boundary.

Extract complete production methods with AST to avoid platform bootstrap and
TT imports. Tensor operations are real CPU torch/numpy; device collaborators
and sampling are inert. Run with ``python -m unittest discover``.
"""

import ast
import copy
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins/vllm-tt-plugin/src/vllm_tt_plugin"


def extract(path, class_name, method_names, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


GLOBALS = {
    "np": np,
    "copy": copy,
    "torch": torch,
    "TTModelInput": N,
    "slice_tt_sampling_params": lambda *args: N(),
    "has_structured_outputs": lambda *args: False,
    "itertools": itertools,
    "CachedRequestData": N,
    "ModelRunnerOutput": N,
}
for node in ast.parse((PLUGIN / "input_batch.py").read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "clone_torch_generator",
        "preempt_cached_request_state",
        "restore_request_rng",
        "acknowledge_request_outputs",
    }:
        module = ast.Module(
            [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], []
        )
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(PLUGIN / "input_batch.py"),
                "exec",
            ),
            GLOBALS,
        )
Runner = extract(
    PLUGIN / "model_runner.py",
    "TTModelRunner",
    {
        "_prepare_model_inputs",
        "_update_states",
        "_build_host_generators",
        "_build_chunked_prefill_output",
    },
    GLOBALS,
)
Scheduler = extract(
    ROOT / "vllm/v1/core/sched/scheduler.py",
    "Scheduler",
    {"_make_cached_request_data", "_update_after_schedule"},
    GLOBALS,
)


def request(prompt=8192, outputs=5, computed=0, req_id="replay", placeholders=0):
    # Worker CachedRequestState.num_tokens is prompt + len(output_token_ids).
    tokens = list(range(prompt + outputs))
    return N(
        request_id=req_id,
        req_id=req_id,
        prompt_token_ids=tokens[:prompt],
        output_token_ids=tokens[prompt:],
        generator=None,
        all_token_ids=tokens,
        num_prompt_tokens=prompt,
        num_tokens=len(tokens),
        num_output_tokens=outputs,
        num_computed_tokens=computed,
        num_output_placeholders=placeholders,
        use_structured_output=False,
        has_encoder_inputs=False,
        is_prefill_chunk=computed < len(tokens),
    )


def runner():
    return N(
        model=N(),
        model_config=N(is_multimodal_model=False),
        max_num_blocks_per_req=1,
        tt_data_parallel_size=1,
        _decode_layout_changed_since_last_decode=True,
        _chunked_prefill_req_ids=set(),
        requests={},
        _block_tables_per_layer=lambda tables: tables,
        _alloc_prefill_state_slots=lambda ids: list(range(len(ids))),
        _decode_state_slot_remap=lambda ids: None,
        check_perform_device_sampling=lambda **kw: True,
        _build_host_generators=Runner._build_host_generators,
    )


def prepare(runner, requests, counts, *, new=(), resumed=(), scheduler_requests=None):
    req_ids = [r.request_id for r in requests]
    num_sched = dict(zip(req_ids, counts))
    scheduler = N(
        use_pp=False,
        scheduler_config=N(async_scheduling=True),
        prev_step_scheduled_req_ids=set(req_ids) - set(resumed),
    )
    scheduler_requests = requests if scheduler_requests is None else scheduler_requests
    cached = Scheduler._make_cached_request_data(
        scheduler,
        [r for r in scheduler_requests if r.request_id not in {*new, *resumed}],
        [r for r in scheduler_requests if r.request_id in resumed],
        num_sched,
        {},
        {rid: N(get_block_ids=lambda **kw: None) for rid in req_ids},
    )
    cached.num_reqs = len(cached.req_ids)
    width = max(r.num_tokens for r in requests)
    token_ids = torch.zeros(len(requests), width, dtype=torch.int32)
    for row, r in enumerate(requests):
        token_ids[row, : r.num_tokens] = torch.tensor(r.all_token_ids)
    # These fields follow InputBatch.add_request's prompt/output copy contract;
    # computed positions and cached metadata come from the real scheduler method.
    batch = N(
        num_reqs=len(requests),
        req_ids=req_ids,
        req_id_to_index=dict(zip(req_ids, range(len(requests)))),
        num_computed_tokens_cpu=np.array([r.num_computed_tokens for r in requests]),
        num_prompt_tokens=np.array([r.num_prompt_tokens for r in requests]),
        num_tokens=np.array([r.num_tokens for r in requests]),
        token_ids_cpu_tensor=token_ids,
        sampling=N(
            bad_words_token_ids={},
            logitsprocs=None,
            generators={
                i: torch.Generator().manual_seed(10 + i) for i in range(len(requests))
            },
        ),
        advanced_generators=[],
        no_penalties=True,
        no_allowed_token_ids=True,
        max_num_logprobs=None,
        block_tables_for_rows=lambda rows, width: [torch.zeros(len(rows), width)],
        reset_slot_remap=lambda: None,
    )
    batch.advance_generators = batch.advanced_generators.extend
    runner.input_batch = batch
    runner.tt_per_lane_max_num_seqs = len(requests)
    runner.requests.update({r.request_id: r for r in requests})
    output = N(
        total_num_scheduled_tokens=sum(counts),
        num_scheduled_tokens=num_sched,
        scheduled_new_reqs=[N(req_id=rid) for rid in new],
        scheduled_cached_reqs=cached,
    )
    return Runner._prepare_model_inputs(runner, output, None), output


def computed_positions(model_input, row=0):
    if model_input.prompt_lens is None:
        return [int(model_input.input_positions[row])]
    start = int(model_input.input_positions[row])
    end = int(model_input.prompt_lens[row])
    # Token IDs equal their positions, making skipped/misattributed tokens visible.
    assert model_input.input_tokens[row, start:end].tolist() == list(range(start, end))
    return list(range(start, end))


class RecomputeTailTest(unittest.TestCase):
    def replay(self, prompt, outputs, chunk=2048, start=0, resumed=True):
        r = runner()
        req = request(prompt, outputs, computed=start)
        positions = []
        modes = []
        while req.num_computed_tokens < req.num_tokens:
            first = req.num_computed_tokens == start
            count = min(chunk, req.num_tokens - req.num_computed_tokens)
            mi, out = prepare(
                r,
                [req],
                [count],
                resumed=[req.request_id] if first and resumed else [],
                new=[req.request_id] if first and not resumed else [],
            )
            positions.extend(computed_positions(mi))
            modes.append("decode" if mi.prompt_lens is None else "prefill")
            if mi.intermediate_prefill_mask is not None:
                self.assertEqual(
                    mi.intermediate_prefill_mask.tolist(),
                    [req.num_computed_tokens + count < req.num_tokens],
                )
            out.has_structured_output_requests = False
            scheduler = N(requests={req.request_id: req}, finished_req_ids=set())
            Scheduler._update_after_schedule(scheduler, out)
        self.assertEqual(positions, list(range(start, req.num_tokens)))
        # Once replay ends, its sampled output makes the following ordinary decode.
        req.all_token_ids.append(req.num_tokens)
        req.num_tokens += 1
        req.num_output_tokens += 1
        mi, _ = prepare(r, [req], [1])
        self.assertIsNone(mi.prompt_lens)
        self.assertEqual(computed_positions(mi), [req.num_tokens - 1])
        return modes

    def test_original_8192_plus_five_replays_every_token(self):
        self.replay(8192, 5)

    def test_prompt_and_chunk_boundaries(self):
        for prompt in (8191, 8192, 8193):
            for outputs in (0, 1, 5, 2049):
                with self.subTest(prompt=prompt, outputs=outputs):
                    self.replay(prompt, outputs)

    def test_new_prompt_and_chunked_prefill(self):
        for prompt in (1, 8191, 8192, 8193):
            with self.subTest(prompt=prompt):
                self.replay(prompt, 0, resumed=False)

    def test_single_token_chunks_in_generated_tail(self):
        self.replay(8192, 5, chunk=1, start=8191)

    def test_resume_directly_at_prompt_boundary(self):
        self.replay(8192, 5, start=8192)

    def test_generated_tail_four_then_one_samples_only_at_final_boundary(self):
        r = runner()
        r.check_perform_device_sampling = lambda **kw: False
        applied = []
        r._apply_sampled_tokens_to_state = lambda tokens, req_ids: applied.append(
            req_ids
        )
        req = request(8192, 5, computed=8192)
        positions = []
        for count, is_intermediate in ((4, True), (1, False)):
            mi, _ = prepare(
                r,
                [req],
                [count],
                resumed=[req.request_id] if is_intermediate else [],
            )
            self.assertEqual(mi.intermediate_prefill_mask.tolist(), [is_intermediate])
            positions.extend(computed_positions(mi))
            batch = r.input_batch
            original = batch.sampling.generators[0]
            before = original.get_state().clone()
            sampled_generator = mi.generators_list[0][0]
            self.assertEqual(sampled_generator is original, not is_intermediate)
            torch.rand(1, generator=sampled_generator)
            self.assertEqual(torch.equal(before, original.get_state()), is_intermediate)
            self.assertEqual(batch.advanced_generators, [] if is_intermediate else [0])
            output = Runner._build_chunked_prefill_output(
                r,
                [req.request_id],
                torch.tensor([[123]]),
                None,
                mi.intermediate_prefill_mask.numpy(),
            )
            self.assertEqual(
                output.sampled_token_ids, [[]] if is_intermediate else [[123]]
            )
            self.assertEqual(applied, [] if is_intermediate else [[req.request_id]])
            req.num_computed_tokens += count
        self.assertEqual(positions, [8192, 8193, 8194, 8195, 8196])

    def test_final_single_token_replay_with_other_prefill(self):
        r = runner()
        req = request(8192, 1, computed=6144)
        prepare(r, [req], [2048], resumed=[req.request_id])
        req.num_computed_tokens = 8192
        new_req = request(128, 0, req_id="new")
        mi, _ = prepare(r, [req, new_req], [1, 128], new=["new"])
        self.assertEqual(computed_positions(mi, 0), [8192])
        self.assertEqual(computed_positions(mi, 1), list(range(128)))

    def test_real_decode_mixed_with_prefill_is_rejected(self):
        r = runner()
        decode = request(8192, 5, computed=8196, req_id="decode")
        new = request(128, 0, req_id="new")
        with self.assertRaisesRegex(AssertionError, "Prefill batch"):
            prepare(r, [decode, new], [1, 128], new=["new"])

    def test_continuation_survives_pause_but_clears_on_finish_or_preempt(self):
        for event in ("pause", "finish", "preempt"):
            with self.subTest(event=event):
                r = runner()
                req = request(8192, 5, computed=6144)
                prepare(r, [req], [2048], resumed=[req.request_id])
                self.assertEqual(r._chunked_prefill_req_ids, {req.request_id})
                r._release_dead_state_slots = lambda output: None
                r.encoder_cache = {}
                batch = r.input_batch
                batch.remove_request = (
                    lambda rid, batch=batch: batch.req_id_to_index.pop(rid, None)
                )
                batch.condense = lambda indices: None
                batch.refresh_logitsprocs = lambda: None
                output = N(
                    finished_req_ids={req.request_id} if event == "finish" else set(),
                    preempted_req_ids={req.request_id} if event == "preempt" else None,
                    free_encoder_mm_hashes=[],
                    num_scheduled_tokens={},
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=N(req_ids=[]),
                )
                Runner._update_states(r, output)
                self.assertEqual(
                    r._chunked_prefill_req_ids,
                    {req.request_id} if event == "pause" else set(),
                )
                req.num_computed_tokens = 8192 if event == "pause" else 0
                if event == "finish":
                    req = request(128, 0, req_id=req.request_id)
                mi, _ = prepare(
                    r,
                    [req],
                    [5 if event == "pause" else 128 if event == "finish" else 2048],
                    resumed=[req.request_id] if event == "preempt" else [],
                    new=[req.request_id] if event == "finish" else [],
                )
                self.assertIsNotNone(mi.prompt_lens)

    def test_ordinary_decode_with_async_placeholders(self):
        for placeholders in (0, 1, 2):
            with self.subTest(placeholders=placeholders):
                r = runner()
                computed = 8196 + placeholders
                scheduler_req = request(
                    8192,
                    5,
                    computed=computed,
                    placeholders=placeholders,
                )
                req = request(8192, 5 + placeholders, computed=computed)
                # Async scheduler metadata includes placeholders, while the worker
                # has the actual tokens delivered by previous device steps.
                self.assertEqual(
                    scheduler_req.num_tokens + placeholders - computed,
                    1,
                )
                mi, out = prepare(r, [req], [1], scheduler_requests=[scheduler_req])
                self.assertEqual(
                    out.scheduled_cached_reqs.num_output_tokens, [5 + placeholders]
                )
                self.assertIsNone(mi.prompt_lens)
                self.assertEqual(computed_positions(mi), [computed])

    def test_no_tt_or_vllm_imports(self):
        self.assertFalse(
            any(
                name == "ttnn" or name.startswith(("ttnn.", "vllm", "vllm_tt_plugin"))
                for name in sys.modules
            )
        )


if __name__ == "__main__":
    unittest.main()
