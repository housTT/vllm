# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which model types keep token-chunked prefill, and what is forced off."""

from types import SimpleNamespace

import pytest

# Importing the platform module reenters vLLM's platform-plugin bootstrap, which
# resolves against a half-built ``vllm`` and fails unless vLLM has already
# finished importing itself. This import has to stay first.
import vllm  # noqa: F401  # isort: skip

from vllm_tt_plugin.platform import _apply_chunked_prefill_policy


def _vllm_config(
    *,
    model_type: str,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 16384,
    long_prefill_token_threshold: int = 512,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            disable_chunked_mm_input=False,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            max_model_len=max_model_len,
        ),
    )


def test_gemma4_keeps_chunked_prefill_and_its_token_budget():
    config = _vllm_config(model_type="gemma4")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 2048
    assert config.scheduler_config.long_prefill_token_threshold == 512
    assert config.scheduler_config.disable_chunked_mm_input is True


def test_unified_gemma4_checkpoint_also_keeps_chunked_prefill():
    config = _vllm_config(model_type="gemma4_unified")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is True


@pytest.mark.parametrize("starting_budget", [512, 1024, 2048, 262144])
def test_ornith_qwen3_5_moe_pins_exact_model_chunk_boundaries(starting_budget):
    config = _vllm_config(
        model_type="qwen3_5_moe",
        max_num_batched_tokens=starting_budget,
        long_prefill_token_threshold=512,
    )

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 8192
    assert config.scheduler_config.long_prefill_token_threshold == 2048
    assert config.scheduler_config.max_num_partial_prefills == 4
    assert config.scheduler_config.max_long_partial_prefills == 4
    assert config.scheduler_config.disable_chunked_mm_input is True


@pytest.mark.parametrize("starting_budget, expected_budget", [(2048, 16384), (32768, 32768)])
def test_ornith_without_chunking_can_admit_a_full_prompt(starting_budget, expected_budget):
    config = _vllm_config(
        model_type="qwen3_5_moe",
        enable_chunked_prefill=False,
        max_num_batched_tokens=starting_budget,
        max_model_len=16384,
    )

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == expected_budget
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_other_model_type_loses_chunked_prefill_and_gets_a_full_prompt_budget():
    config = _vllm_config(model_type="llama")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 16384


def test_unsplit_prefill_leaves_chunked_mm_input_enabled():
    # vLLM raises outright when this is set and one mm item is larger than
    # max_num_batched_tokens, e.g. a VL model pinned to a short max_model_len.
    # With prefill never split the flag is inert, so it must stay off.
    config = _vllm_config(
        model_type="qwen2_5_vl", max_num_batched_tokens=2048, max_model_len=2048
    )

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.disable_chunked_mm_input is False


def test_other_model_type_zeroes_the_long_prefill_threshold():
    # The base scheduler applies this cap before it consults
    # enable_chunked_prefill, so leaving it set would still split a prefill.
    config = _vllm_config(model_type="llama", enable_chunked_prefill=False)

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Chunked prefill was already off, so the token budget is left alone.
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_token_budget_is_left_alone_when_it_already_covers_the_model_len():
    config = _vllm_config(model_type="llama", max_num_batched_tokens=32768)

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.max_num_batched_tokens == 32768
