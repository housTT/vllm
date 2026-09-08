# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for bad-word history; no TTNN import or device execution."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).resolve().parents[1]
VLLM = PLUGIN.parents[1]


def _history_block():
    """Compile the runner's history preparation without its TTNN imports."""
    path = PLUGIN / "src/vllm_tt_plugin/model_runner.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_model_inputs"
    )
    index = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "prompt_tokens"
            for target in node.targets
        )
    )
    body = method.body[index : index + 3]
    assert isinstance(body[-1], ast.If)
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


def _prepare_history(*, no_penalties=True, bad_words=None, is_prompt=False):
    calls = []
    # Source rows 0 and 2 deliberately have different output histories.
    outputs = torch.tensor([[99], [42], [7]], dtype=torch.int32)
    prompts = torch.tensor([[10], [20], [30]], dtype=torch.int32)

    def gather(name, tensor, rows):
        calls.append(name)
        return tensor[rows].clone()

    batch = SimpleNamespace(
        no_penalties=no_penalties,
        sampling=SimpleNamespace(bad_words_token_ids=bad_words or {}),
        max_num_reqs=4,
        make_prompt_token_ids_tensor=lambda rows: gather("prompt", prompts, rows),
        make_output_token_ids_tensor=lambda rows: gather("output", outputs, rows),
    )
    namespace = {
        "input_batch": batch,
        "is_prompt": is_prompt,
        "req_indices": [2, 0],
        "self": SimpleNamespace(tt_data_parallel_size=1),
        "torch": torch,
    }
    exec(_history_block(), namespace)
    return namespace["output_tokens"], calls


def test_multitoken_bad_words_preserve_prefix_and_request_rows():
    history, calls = _prepare_history(bad_words={0: [[7, 11]], 2: [[7, 11]]})
    assert history is not None, "bad words need output history with neutral penalties"
    assert calls == ["prompt", "output"]
    assert history.tolist() == [[7], [99], [-1], [-1]]
    apply_bad_words = runpy.run_path(str(VLLM / "vllm/v1/sample/ops/bad_words.py"))[
        "apply_bad_words"
    ]
    logits = torch.zeros(2, 128)
    apply_bad_words(logits, {0: [[7, 11]], 1: [[7, 11]]}, history[:2].tolist())
    assert torch.isneginf(logits[0, 11])
    assert torch.isfinite(logits[1, 11])


def test_ordinary_device_sampling_does_not_copy_histories():
    history, calls = _prepare_history()
    assert history is None
    assert calls == []


def test_prefill_does_not_copy_decode_history():
    history, calls = _prepare_history(bad_words={0: [[7, 11]]}, is_prompt=True)
    assert history is None
    assert calls == []


def test_penalty_history_preserves_existing_row_selection_and_padding():
    history, calls = _prepare_history(no_penalties=False)
    assert history.tolist() == [[7], [99], [-1], [-1]]
    assert calls == ["prompt", "output"]


def _assert_no_bad_words():
    source = PLUGIN / "tests/tt/test_host_only_params.py"
    helper = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_assert_no_bad_word_sequences"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[helper.name]


@pytest.mark.parametrize(
    "tokens,banned", [([71, 14556, 4638], [[14556]]), ([99, 7, 11], [[7, 11]])]
)
def test_serving_assertion_rejects_actual_banned_token_sequence(tokens, banned):
    with pytest.raises(AssertionError, match="Banned token sequence"):
        _assert_no_bad_words()(tokens, banned)


def test_serving_assertion_distinguishes_allowed_alternate_segmentation():
    # Actual Ornith serving counterexample: "h" + "ello" versus "hello".
    _assert_no_bad_words()([71, 4638], [[14556], [23066]])
    _assert_no_bad_words()([7, 9, 11], [[7, 11]])
