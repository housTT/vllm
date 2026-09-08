# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU controls for canonical first-token checks; no plugin/TTNN imports."""

import ast
import asyncio
import random
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).resolve().parents[1]


def _response(text, token_ids, finish_reason="length"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                text=text,
                finish_reason=finish_reason,
                model_dump=lambda: {"text": text, "token_ids": token_ids},
            )
        ]
    )


def _canonical(batch_responses, eos_token_id=248046):
    """Compile the actual tests with a CPU stand-in only for HTTP requests."""
    utils = runpy.run_path(str(PLUGIN / "tests/tt/utils.py"))
    calls = []

    def run_batch(server, model, configs, return_full_response=False):
        calls.append(configs)
        responses = batch_responses[len(calls) - 1]
        if return_full_response:
            return responses
        return [response.choices[0].text for response in responses]

    namespace = {
        **utils,
        "run_concurrent_batch": run_batch,
        "replace": replace,
        "pytest": pytest,
        "random": random,
        "cached_get_tokenizer": lambda model: SimpleNamespace(
            eos_token_id=eos_token_id
        ),
    }
    source = PLUGIN / "tests/tt/test_seeding_and_variety.py"
    body = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["TestSeedingAndVariety"](), calls


def test_different_actual_first_tokens_with_same_first_character_pass():
    responses = [_response(f" item{i}", [100 + i, 500]) for i in range(7)]
    canonical, calls = _canonical([responses])
    canonical.test_temperature_varied_in_batch(None, "test-model", 7)
    assert len(calls) == 1 and len(calls[0]) == 7
    assert all(config.return_token_ids for config in calls[0])
    assert all(
        config.prompt == "Random letter: "
        and config.temperature == 5
        and config.max_tokens == 10
        and config.seed is None
        and config.top_k is None
        for config in calls[0]
    )


def test_same_actual_first_token_fails_despite_varied_later_text():
    responses = [_response(f" shared item{i}", [100, 500 + i]) for i in range(7)]
    canonical, _ = _canonical([responses])
    with pytest.raises(AssertionError, match="first tokens should vary"):
        canonical.test_temperature_varied_in_batch(None, "test-model", 7)


@pytest.mark.parametrize("same_first_tokens", [False, True])
def test_topk_keeps_pairwise_first_token_and_full_output_checks(same_first_tokens):
    greedy = [_response(" stable", [42, 50]) for _ in range(7)]
    first = greedy + [_response(f" sample{i}", [100 + i, 500]) for i in range(8)]
    offset = 0 if same_first_tokens else 10
    second = greedy + [
        _response(f" changed{i}", [100 + i + offset, 600]) for i in range(8)
    ]
    canonical, _ = _canonical([first, second])
    if same_first_tokens:
        with pytest.raises(
            AssertionError, match="first tokens should vary when re-ran"
        ):
            canonical.test_topk(None, "test-model", 15)
    else:
        canonical.test_topk(None, "test-model", 15)


@pytest.mark.parametrize("changed_first_seed_token", [False, True])
def test_seed_checks_preserve_full_text_and_actual_token_identity(
    changed_first_seed_token,
):
    first = [_response(" stable", [42])] * 3 + [
        _response(f" seed{i}", [100 + i]) for _ in range(2) for i in range(3)
    ]
    second = list(first)
    if changed_first_seed_token:
        second[3] = _response(" seed0", [999])
    canonical, _ = _canonical([first, second])
    if changed_first_seed_token:
        with pytest.raises(AssertionError, match="Seeded first tokens should match"):
            canonical.test_seeding(None, "test-model", 9)
    else:
        canonical.test_seeding(None, "test-model", 9)


@pytest.mark.parametrize("token_ids", [None, []])
def test_missing_actual_token_metadata_is_rejected(token_ids):
    responses = [_response(f" item{i}", token_ids) for i in range(7)]
    canonical, _ = _canonical([responses])
    with pytest.raises(AssertionError, match="generated token IDs are missing"):
        canonical.test_temperature_varied_in_batch(None, "test-model", 7)


@pytest.mark.parametrize(
    "token_ids,finish_reason",
    [([100], "stop"), ([248046], "length"), ([248046], None)],
)
def test_empty_completion_without_terminal_eos_evidence_is_rejected(
    token_ids, finish_reason
):
    canonical, _ = _canonical([[_response("", token_ids, finish_reason)] * 7])
    with pytest.raises(AssertionError, match="without terminal EOS evidence"):
        canonical.test_temperature_varied_in_batch(None, "test-model", 7)


def test_topk_accepts_valid_immediate_eos_and_retains_variety_checks():
    greedy = [_response(" stable", [42, 50]) for _ in range(7)]
    first = greedy + [_response(f" sample{i}", [100 + i, 500]) for i in range(8)]
    second = greedy + [_response(f" changed{i}", [200 + i, 600]) for i in range(8)]
    first[9] = _response("", [248046], finish_reason="stop")
    second[14] = _response("", [248046], finish_reason="stop")
    canonical, _ = _canonical([first, second])
    canonical.test_topk(None, "test-model", 15)


def test_identical_immediate_eos_outputs_still_fail_variety():
    canonical, _ = _canonical([[_response("", [248046], "stop")] * 7])
    with pytest.raises(AssertionError, match="outputs should vary"):
        canonical.test_temperature_varied_in_batch(None, "test-model", 7)


def test_empty_completion_uses_the_configured_tokenizer_eos():
    responses = [_response("", [777], "stop")] + [
        _response(f" item{i}", [100 + i]) for i in range(6)
    ]
    canonical, _ = _canonical([responses], eos_token_id=777)
    canonical.test_temperature_varied_in_batch(None, "test-model", 7)


def test_missing_completion_text_is_rejected():
    canonical, _ = _canonical([[_response(None, [248046], "stop")] * 7])
    with pytest.raises(AssertionError, match="completion text is missing"):
        canonical.test_temperature_varied_in_batch(None, "test-model", 7)


@pytest.mark.parametrize("enabled", [False, True])
def test_completion_token_metadata_is_explicit_and_optional(enabled):
    utils = runpy.run_path(str(PLUGIN / "tests/tt/utils.py"))
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _response(" item", [100])

    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    config = utils["RequestConfig"](prompt="test", return_token_ids=enabled)
    text = asyncio.run(utils["send_request"](client, "test-model", config))
    assert text == " item"
    assert captured["extra_body"] == ({"return_token_ids": True} if enabled else None)
