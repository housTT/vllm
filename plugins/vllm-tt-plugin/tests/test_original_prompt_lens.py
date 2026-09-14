# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU controls for the original prompt boundary in recompute payloads.

Complete production definitions are extracted without importing TT/platform
modules. Tensor operations and payload dataclasses are real; device submission
is recorded and never executed.
"""

import ast
import dataclasses
import itertools
import pickle
import sys
import typing
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch

import test_accepted_history_rng as history
import test_recompute_tail as tail

np, torch = tail.np, tail.torch
G = dict(tail.GLOBALS)
G.update(
    dataclass=dataclasses.dataclass,
    fields=dataclasses.fields,
    cast=typing.cast,
    __name__=__name__,
    SEED_NONE_SENTINEL=-1,
)
for node in ast.parse((tail.PLUGIN / "model_input.py").read_text()).body:
    if isinstance(node, ast.ClassDef):
        module = ast.Module(
            [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], []
        )
        exec(compile(ast.fix_missing_locations(module), "model_input.py", "exec"), G)
TTModelInput = G["TTModelInput"]
TTSamplingParams = G["TTSamplingParams"]
Runner = tail.extract(
    tail.PLUGIN / "model_runner.py",
    "TTModelRunner",
    {"concat_dp_model_inputs", "_merge_dp_prefill_slots", "submit_prefill"},
    G,
)
Lane = tail.extract(
    tail.PLUGIN / "input_batch.py", "TTLaneInputBatch", {"_build_prefill_input"}, G
)


def sampling(n):
    return TTSamplingParams(
        **{
            field.name: torch.zeros(
                n,
                dtype=torch.int64
                if field.name in {"top_k", "seed", "num_logprobs"}
                else torch.float32,
            )
            for field in dataclasses.fields(TTSamplingParams)
        }
    )


def payload(prompts, ends=None):
    ends = prompts if ends is None else ends
    n = len(prompts)
    reqs = [
        tail.request(prompt=p, outputs=max(0, e - p), req_id=f"r{i}")
        for i, (p, e) in enumerate(zip(prompts, ends))
    ]
    mi, _ = tail.prepare(tail.runner(), reqs, ends, new=[r.req_id for r in reqs])
    mi.tt_sampling_params = sampling(n)
    return TTModelInput(**vars(mi))


def merge(inputs):
    r = N(
        max_num_blocks_per_req=4,
        _num_kv_cache_groups=1,
        tt_per_lane_max_num_seqs=8,
        model_config=N(is_multimodal_model=False),
        _block_tables_per_layer=lambda groups: groups,
    )
    r._merge_dp_prefill_slots = lambda values: Runner._merge_dp_prefill_slots(r, values)
    return Runner.concat_dp_model_inputs(r, inputs, False, None, False)


class OriginalPromptLensTest(unittest.TestCase):
    def test_boundary_stays_fixed_across_recompute_chunks(self):
        for prompt, outputs, chunk in [
            (29, 5, 8),
            (8192, 5, 2048),
            (31, 1, 1),
            (1, 65, 32),
        ]:
            with self.subTest(prompt=prompt, outputs=outputs, chunk=chunk):
                r = tail.runner()
                req = tail.request(prompt=prompt, outputs=outputs)
                while req.num_computed_tokens < req.num_tokens:
                    start = req.num_computed_tokens
                    size = min(chunk, req.num_tokens - start)
                    mi, _ = tail.prepare(
                        r, [req], [size], resumed=[req.req_id] if start == 0 else []
                    )
                    self.assertEqual(mi.original_prompt_lens, [prompt])
                    self.assertEqual(mi.prompt_lens.tolist(), [start + size])
                    self.assertEqual(
                        tail.computed_positions(mi), list(range(start, start + size))
                    )
                    req.num_computed_tokens += size

    def test_boundary_survives_authoritative_history_and_late_async_outputs(self):
        seen = []
        original = history.Runner._prepare_model_inputs

        def record(runner, scheduled, grammar):
            mi = original(runner, scheduled, grammar)
            if mi.prompt_lens is not None:
                self.assertEqual(mi.original_prompt_lens, [8192])
                seen.append(mi.original_prompt_lens)
            else:
                self.assertIsNone(mi.original_prompt_lens)
            return mi

        with patch.object(history.Runner, "_prepare_model_inputs", record):
            history.AcceptedHistoryTest(
                "test_stale_history_before_or_after_resume_and_full_tail_replay"
            ).test_stale_history_before_or_after_resume_and_full_tail_replay()
        self.assertEqual(len(seen), 66)

    def test_mixed_rows_and_detached_metadata(self):
        r = tail.runner()
        a = tail.request(prompt=29, outputs=5, computed=24, req_id="a")
        b = tail.request(prompt=8192, outputs=5, computed=8196, req_id="b")
        mi, _ = tail.prepare(r, [b, a], [1, 8], resumed=["a", "b"])
        self.assertEqual(mi.original_prompt_lens, [8192, 29])
        self.assertEqual(mi.prompt_lens.tolist(), [8197, 32])
        self.assertEqual(mi.intermediate_prefill_mask.tolist(), [False, True])
        r.input_batch.num_prompt_tokens[:] = 7
        self.assertEqual(mi.original_prompt_lens, [8192, 29])

    def test_padded_decode_has_no_prefill_boundary(self):
        r = tail.runner()
        r.model.tt_supported_decode_batch_sizes = [1, 32]
        req = tail.request(prompt=29, outputs=5, computed=33)
        other = tail.request(prompt=31, outputs=2, computed=32, req_id="other")
        _, scheduled = tail.prepare(r, [req, other], [1, 1])
        r.tt_per_lane_max_num_seqs = 32
        r._sampling_params_for_padded_decode = lambda params, rows, n: sampling(n)
        mi = tail.Runner._prepare_model_inputs(r, scheduled, None)
        self.assertEqual(mi.input_tokens.shape[0], 32)
        self.assertTrue(torch.all(mi.input_positions[2:] == -1))
        self.assertIsNone(mi.original_prompt_lens)
        self.assertIsNone(mi.prompt_lens)

    def test_lane_order_empty_lanes_and_padding(self):
        for rows in itertools.permutations([7, 1, 4]):
            batch = N(
                num_computed_tokens_cpu=np.array([0, 24, 0, 0, 8196, 0, 0, 0]),
                num_prompt_tokens=np.array([3, 29, 5, 6, 8192, 8, 9, 31]),
                num_tokens=np.array([3, 34, 5, 6, 8197, 8, 9, 31]),
                req_ids=[f"r{i}" for i in range(8)],
                token_ids_cpu_tensor=torch.arange(8 * 8200).reshape(8, 8200),
                max_num_reqs=8,
                max_num_logprobs=None,
                no_penalties=True,
                slot_block_tables=lambda *a, **kw: [torch.zeros(3, 4)],
                slot_sampling_params=lambda rows: sampling(len(rows)),
                slot_grammar_bitmask=lambda *a: None,
            )
            r = N(
                max_num_blocks_per_req=4,
                requests={},
                model_config=N(is_multimodal_model=False),
                check_perform_device_sampling=lambda **kw: True,
                _block_tables_per_layer=lambda groups: groups,
            )
            plan = N(
                input_rows=rows,
                batch_size_per_dp=[1, 0, 2, 0],
                prefill_empty_slots=[19, 2, 11],
            )
            mi = Lane._build_prefill_input(
                batch,
                r,
                N(num_scheduled_tokens={"r7": 31, "r1": 8, "r4": 1}),
                None,
                plan,
            )
            self.assertEqual(
                mi.original_prompt_lens, batch.num_prompt_tokens[list(rows)].tolist()
            )
            self.assertEqual(
                mi.input_tokens[:, 0].tolist(), [row * 8200 for row in rows]
            )
            self.assertEqual(mi.unpadded_batch_size, [1, 0, 2, 0])
            self.assertEqual(mi.prefill_empty_slots, [19, 2, 11])
            self.assertEqual(len(mi.original_prompt_lens), 3)

    def test_dp_gather_padding_and_empty_ranks(self):
        a, b = payload([29, 31], [34, 32]), payload([8192], [8197])
        # Object gather transports the whole prefill payload, unlike packed decode.
        inputs = pickle.loads(pickle.dumps([None, a, None, b]))
        result = merge(inputs)
        self.assertEqual(result.original_prompt_lens, [29, 31, 8192])
        self.assertEqual(result.prompt_lens.tolist(), [34, 32, 8197])
        self.assertEqual(result.input_tokens.shape, (3, 8197))
        self.assertEqual(result.unpadded_batch_size, [0, 2, 0, 1])
        self.assertEqual(result.prefill_empty_slots, [8, 9, 24])
        self.assertEqual(result.input_tokens[0, 34:].count_nonzero().item(), 0)

    def test_dp_legacy_payloads_and_missing_metadata(self):
        a = payload([29])
        legacy = dataclasses.replace(a, original_prompt_lens=None)
        self.assertIsNone(merge([None, legacy]).original_prompt_lens)
        with self.assertRaisesRegex(ValueError, "original prompt length per row"):
            merge([a, legacy])
        with self.assertRaisesRegex(ValueError, "original prompt length per row"):
            merge([dataclasses.replace(a, original_prompt_lens=[])])

    def test_prepared_input_replacement_preserves_boundary(self):
        mi = payload([29], [34])
        replaced = dataclasses.replace(mi, grammar_bitmask=[torch.ones(1, 1)])
        self.assertEqual(replaced.original_prompt_lens, [29])

    def test_capability_opt_in_preserves_legacy_kwargs(self):
        mi = payload([29], [34])
        recorded = []
        model = N(prefill_forward=lambda **kw: recorded.append(kw))
        r = N(
            model=model,
            kv_caches="cache",
            trace_mode="none",
            request_specific_rope=False,
            tt_per_lane_max_num_seqs=8,
        )
        Runner.submit_prefill(r, mi, [1])
        model.supports_original_prompt_lens = False
        Runner.submit_prefill(r, mi, [1])
        model.supports_original_prompt_lens = True
        Runner.submit_prefill(r, mi, [1])
        self.assertEqual(recorded[0].keys(), recorded[1].keys())
        self.assertNotIn("original_prompt_lens", recorded[0])
        self.assertEqual(recorded[2].pop("original_prompt_lens"), [29])
        for name, value in recorded[0].items():
            for other in recorded[1:]:
                if name in {"empty_slots", "sampling_params"}:
                    self.assertEqual(other[name], value)
                else:
                    self.assertIs(other[name], value)

    def test_capable_submission_rejects_missing_or_misaligned_boundary(self):
        recorded = []
        model = N(
            supports_original_prompt_lens=True,
            prefill_forward=lambda **kw: recorded.append(kw),
        )
        r = N(
            model=model,
            kv_caches=None,
            trace_mode="none",
            request_specific_rope=False,
            tt_per_lane_max_num_seqs=8,
        )
        for value in (None, [], [29, 31]):
            mi = dataclasses.replace(payload([29]), original_prompt_lens=value)
            with self.assertRaisesRegex(ValueError, "per execution row"):
                Runner.submit_prefill(r, mi, [1])
        self.assertEqual(recorded, [])
        Runner.submit_prefill(r, payload([0], [1]), [1])
        self.assertEqual(recorded.pop()["original_prompt_lens"], [0])
        model.supports_original_prompt_lens = False
        Runner.submit_prefill(
            r, dataclasses.replace(payload([29]), original_prompt_lens=None), [1]
        )
        self.assertEqual(len(recorded), 1)
        self.assertNotIn("original_prompt_lens", recorded[0])

    def test_no_tt_imports(self):
        self.assertFalse(
            any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)
        )


if __name__ == "__main__":
    unittest.main()
