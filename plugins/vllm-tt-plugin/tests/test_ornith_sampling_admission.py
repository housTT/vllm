# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only admission tests; model registration/loading are replaced by stubs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Finish vLLM bootstrap before importing the platform plugin.
import vllm  # noqa: F401  # isort: skip

from vllm_tt_plugin import platform as tt_platform

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.models.registry import ModelRegistry
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

HOST_PARAMS = [
    pytest.param({"min_p": 0.2}, "min_p", id="min_p"),
    pytest.param({"logit_bias": {1: 0.1}}, "logit_bias", id="logit_bias"),
    pytest.param({"allowed_token_ids": [1]}, "allowed_token_ids", id="allowlist"),
    pytest.param({"bad_words": ["word"]}, "bad_words", id="bad_words"),
    pytest.param({"min_tokens": 5}, "min_tokens", id="min_tokens"),
    pytest.param(
        {"structured_outputs": StructuredOutputsParams(json_object=True)},
        "structured_outputs",
        id="structured_outputs",
    ),
    pytest.param({"logprobs": 0}, "logprobs", id="logprobs_0"),
    pytest.param({"logprobs": 1}, "logprobs", id="logprobs_1"),
    pytest.param({"logprobs": 5}, "logprobs", id="logprobs_5"),
]


@pytest.fixture
def platform(monkeypatch):
    class TestPlatform(tt_platform.TTPlatform):
        pass

    class ModelStub:
        model_capabilities = {
            "supports_sample_on_device": True,
            "supports_async_decode": True,
        }

    monkeypatch.delenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", raising=False)
    monkeypatch.setattr(tt_platform, "register_tt_models", Mock())
    monkeypatch.setattr(tt_platform, "_install_tt_harmony_truncation_patch", Mock())
    monkeypatch.setattr(
        ModelRegistry,
        "get_supported_archs",
        lambda: ["TTOrnithForCausalLM", "TTQwen3_5ForConditionalGeneration"],
    )
    monkeypatch.setattr(
        loader_utils,
        "get_model_architecture",
        lambda config: (ModelStub, config.hf_config.architectures[0]),
    )
    return TestPlatform


def configure(platform, arch="OrnithForCausalLM", **tt_options):
    config = SimpleNamespace(
        additional_config={"tt": {"sample_on_device_mode": "all", **tt_options}},
        speculative_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            worker_cls="auto",
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[arch] if isinstance(arch, str) else arch,
                model_type="qwen3_5",
            ),
            model="test-model",
            max_logprobs=20,
            max_model_len=2048,
            logits_processors=[],
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            max_num_batched_tokens=2048,
            long_prefill_token_threshold=0,
            async_scheduling=False,
            verify_max_model_len=Mock(),
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    platform.check_and_update_config(config)
    return config


def validate(platform, params):
    platform.validate_request(
        prompt={"prompt_token_ids": [1]},
        params=params,
        processed_inputs={"type": "token", "prompt_token_ids": [1]},
    )


@pytest.mark.parametrize("kwargs, parameter", HOST_PARAMS)
def test_default_ornith_rejects_host_parameters(platform, kwargs, parameter):
    configure(platform)
    params = SamplingParams(**kwargs)
    before = repr(params)
    with pytest.raises(
        VLLMValidationError, match="ORNITH_VLLM_ALLOW_HOST_SAMPLING=1"
    ) as exc:
        validate(platform, params)
    assert exc.value.parameter == parameter
    assert repr(params) == before
    # Rejection must not leave admission unusable for the next valid request.
    validate(platform, SamplingParams(temperature=0.0))


@pytest.mark.parametrize("kwargs, parameter", HOST_PARAMS)
def test_host_compatibility_keeps_parameters_enabled(
    platform, monkeypatch, kwargs, parameter
):
    monkeypatch.setenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", "1")
    configure(platform)
    validate(platform, SamplingParams(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_p": 0.0, "min_tokens": 0, "logit_bias": {}, "bad_words": []},
        {"temperature": 0.0, "min_p": 0.2},
        {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.9,
            "seed": 42,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "repetition_penalty": 1.1,
        },
    ],
)
def test_supported_and_canonicalized_sampling(platform, kwargs):
    configure(platform)
    validate(platform, SamplingParams(**kwargs))


@pytest.mark.parametrize("arch", ["OrnithForCausalLM", "TTOrnithForCausalLM"])
@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_architecture_aliases_and_exact_opt_in(platform, monkeypatch, arch, flag):
    if flag is not None:
        monkeypatch.setenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", flag)
    configure(platform, arch)
    with pytest.raises(VLLMValidationError):
        validate(platform, SamplingParams(min_p=0.2))


def test_configuration_resets_policy_and_captures_environment(platform, monkeypatch):
    configure(platform)
    monkeypatch.setenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", "1")
    with pytest.raises(VLLMValidationError):
        validate(platform, SamplingParams(min_p=0.2))
    configure(platform)
    validate(platform, SamplingParams(min_p=0.2))
    monkeypatch.setenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", "0")
    validate(platform, SamplingParams(min_p=0.2))
    configure(platform)
    with pytest.raises(VLLMValidationError):
        validate(platform, SamplingParams(min_p=0.2))
    configure(platform, "Qwen3_5ForConditionalGeneration")
    for case in HOST_PARAMS:
        validate(platform, SamplingParams(**case.values[0]))


@pytest.mark.parametrize("flag", ["0", "1"])
def test_prompt_logprobs_remain_unsupported(platform, monkeypatch, flag):
    monkeypatch.setenv("ORNITH_VLLM_ALLOW_HOST_SAMPLING", flag)
    configure(platform)
    with pytest.raises(ValueError, match="prompt_logprobs"):
        validate(platform, SamplingParams(prompt_logprobs=1))


def test_unselected_ornith_architecture_does_not_restrict_selected_model(platform):
    configure(platform, ["Qwen3_5ForConditionalGeneration", "OrnithForCausalLM"])
    validate(platform, SamplingParams(min_p=0.2))


def test_rejection_creates_a_400_response_with_parameter(platform):
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    configure(platform)
    with pytest.raises(VLLMValidationError) as exc:
        validate(platform, SamplingParams(min_p=0.2))
    serving = object.__new__(OpenAIServing)
    serving.log_error_stack = False
    response = serving.create_error_response(exc.value)
    assert response.error.code == 400
    assert response.error.param == "min_p"
    assert response.error.type == "BadRequestError"
    assert "ORNITH_VLLM_ALLOW_HOST_SAMPLING=1" in response.error.message
