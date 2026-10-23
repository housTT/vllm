"""Actual-source host tests for authoritative history after async preemption."""
import ast
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
import unittest

POLICY_SOURCE = Path(__file__).resolve().parents[1] / "src/vllm_tt_plugin"
SOURCE = Path(os.environ.get("ORNITH_PREEMPTION_SOURCE", str(POLICY_SOURCE)))
LIVE = POLICY_SOURCE


def extract(path, name, class_name=None, extra=None):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    node = next(
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = {
        "dataclass": dataclass,
        "ClassVar": ClassVar,
        "np": SimpleNamespace(int32="int32"),
        **(extra or {}),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


def update(state, ids, count=None, resumed=True):
    helper = extract(SOURCE / "input_batch.py", "apply_cached_req_state_update")
    kwargs = {}
    if "all_token_ids" in inspect.signature(helper).parameters:
        kwargs = {"all_token_ids": ids, "num_output_tokens": count}
    helper(state, 0 if resumed else 15, ([22],), resumed, **kwargs)


def request(outputs):
    return SimpleNamespace(
        req_id="r",
        prompt_token_ids=[10, 11],
        output_token_ids=list(outputs),
        num_computed_tokens=15,
        block_ids=([3],),
    )


class Array(list):
    dtype = "int32"


class Tokens:
    shape = (1, 1)

    def view(self, _size):
        return self

    def numpy(self):
        return Array([999])


class Batch:
    def __init__(self):
        self.req_id_to_index = {}
        self.num_computed_tokens_cpu = {}
        self.block_table = SimpleNamespace(append_row=lambda *_: None)
        self.added = []

    def remove_request(self, req_id):
        return self.req_id_to_index.pop(req_id, None)

    def add_request(self, state, index=None):
        self.added.append((state.req_id, list(state.output_token_ids)))
        self.req_id_to_index[state.req_id] = 0 if index is None else index

    add_request_to_row = add_request

    def refresh_logitsprocs(self):
        pass

    def condense(self, _indices):
        pass


class History(unittest.TestCase):
    def test_discarded_async_output_removed_on_resume(self):
        pending_type = extract(LIVE / "scheduler.py", "_PendingOutputs")
        apply_token = extract(
            SOURCE / "model_runner.py",
            "_apply_sampled_tokens_to_state",
            "TTModelRunner",
        )
        state = request([101, 102])
        shared = state.output_token_ids
        pending = pending_type()
        pending.record()
        pending.discard_outstanding()
        runner = SimpleNamespace(requests={"r": state}, input_batch=Batch())
        apply_token(runner, Tokens(), req_ids=["r"], request_states=(state,))
        self.assertEqual(shared, [101, 102, 999])
        self.assertTrue(pending.is_next_stale())
        update(state, [10, 11, 101, 102], 2)
        self.assertIs(state.output_token_ids, shared)
        self.assertEqual(shared, [101, 102])
        self.assertEqual(state.num_computed_tokens, 0)
        self.assertEqual(state.block_ids, ([22],))

    def test_authoritative_prefix_missing_tokens_repeated_and_empty_resume(self):
        state = request([101])
        for committed in ([101, 102], [101, 102, 103], []):
            update(state, [10, 11] + committed, len(committed))
            self.assertEqual(state.output_token_ids, committed)
            state.output_token_ids.extend([777, 888])

    def test_unpreempted_update_preserves_history_and_appends_blocks(self):
        state = request([101, 999])
        shared = state.output_token_ids
        update(state, None, resumed=False)
        self.assertIs(state.output_token_ids, shared)
        self.assertEqual(shared, [101, 999])
        self.assertEqual(state.block_ids, ([3, 22],))
        self.assertEqual(state.num_computed_tokens, 15)

    def test_incoherent_authority_rejected_before_mutation(self):
        for ids, count in [
            (None, 2),
            ([10, 11, 101], None),
            ([12, 11, 101], 1),
            ([10, 11, 101], 2),
            ([10, 11], -1),
        ]:
            state = request([101, 999])
            with self.subTest(ids=ids, count=count), self.assertRaises(AssertionError):
                update(state, ids, count)
            self.assertEqual(state.output_token_ids, [101, 999])
            self.assertEqual(state.num_computed_tokens, 15)
            self.assertEqual(state.block_ids, ([3],))

    def test_front_and_lane_actual_callers_reconcile_before_readding(self):
        helper = extract(SOURCE / "input_batch.py", "apply_cached_req_state_update")
        namespace = {"apply_cached_req_state_update": helper}
        generic = extract(
            SOURCE / "model_runner.py", "_update_states", "TTModelRunner", namespace
        )
        lane = extract(
            SOURCE / "input_batch.py", "apply_step_plan", "TTLaneInputBatch", namespace
        )
        for mode in ("front", "lane"):
            state = request([101, 102, 999])
            batch = Batch()
            cached = SimpleNamespace(
                req_ids=["r"],
                num_computed_tokens=[0],
                new_block_ids=[([22],)],
                resumed_req_ids={"r"},
                all_token_ids={"r": [10, 11, 101, 102]},
                num_output_tokens=[2],
            )
            output = SimpleNamespace(
                finished_req_ids=set(),
                preempted_req_ids=set(),
                free_encoder_mm_hashes=[],
                num_scheduled_tokens={"r": 4},
                scheduled_new_reqs=[],
                scheduled_cached_reqs=cached,
            )
            with self.subTest(mode=mode):
                if mode == "front":
                    runner = SimpleNamespace(
                        requests={"r": state},
                        input_batch=batch,
                        encoder_cache={},
                        _release_dead_state_slots=lambda _: None,
                    )
                    generic(runner, output)
                else:
                    lane(
                        batch,
                        output,
                        SimpleNamespace(req_id_to_row={"r": 5}),
                        {"r": state},
                        {},
                    )
                self.assertEqual(batch.added, [("r", [101, 102])])

    def test_actual_resume_requires_async_drain_before_state_rebuild(self):
        # Execute the exact initial drain sequence from execute_model, stopping at build_model_input.
        tree = ast.parse((SOURCE / "model_runner.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "TTModelRunner"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "execute_model"
        )
        start = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "apply_ready_completed_decode_steps"
        )
        stop = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "model_input" for t in n.targets
            )
        )
        events = []
        controller = SimpleNamespace(
            apply_ready_completed_decode_steps=lambda: events.append("apply_ready"),
            can_attempt_steady_decode_from_scheduler=lambda *_: False,
            must_drain_pending_async_steps=lambda candidate: not candidate,
            wait_for_all_pending_async_steps=lambda: events.append("drain"),
        )
        runner = SimpleNamespace(
            async_decode=controller, build_model_input=lambda *_: events.append("build")
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=fn.body[start : stop + 1], type_ignores=[])
                ),
                "actual_execute_prefix",
                "exec",
            ),
            {"self": runner, "scheduler_output": object()},
        )
        self.assertEqual(events, ["apply_ready", "drain", "build"])
        predicate = extract(
            LIVE / "async_decode.py",
            "steady_decode_scheduler_invariants_met",
            "TTAsyncDecodeController",
        )
        self.assertFalse(
            predicate(
                SimpleNamespace(runner=SimpleNamespace()),
                SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"r"}),
                ),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
