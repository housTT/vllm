# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL
from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(**overrides):
    sampling = SimpleNamespace(
        bad_words_token_ids=[],
        presence_penalties_reqs=set(),
        frequency_penalties_reqs=set(),
        repetition_penalties_reqs=set(),
        seed=torch.tensor([SEED_NONE_SENTINEL, SEED_NONE_SENTINEL]),
        top_k=torch.tensor([32, 32]),
        temperature=torch.tensor([1.0, 1.0]),
        has_active_logitsprocs=lambda: False,
    )
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        num_devices=1,
        tt_data_parallel_size=1,
        supports_device_penalties=False,
        supports_device_seeded_sampling=False,
        max_device_top_k=32,
        supports_topk_logprobs=False,
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            no_allowed_token_ids=True,
            max_num_logprobs=None,
            num_reqs=2,
            presence_penalties_reqs=set(),
            frequency_penalties_reqs=set(),
            repetition_penalties_reqs=set(),
            sampling=sampling,
        ),
    )
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _eligible(runner):
    return TTModelRunner.check_perform_device_sampling(
        runner, is_decode=True, has_structured_outputs=False
    )


def test_greedy_and_supported_unseeded_sampling_stay_on_device():
    runner = _runner()
    assert _eligible(runner)
    runner.input_batch.sampling.temperature[:] = 0
    runner.input_batch.sampling.top_k[:] = 100
    assert _eligible(runner)


def test_unsupported_seed_penalty_and_stochastic_top_k_use_host_sampler():
    runner = _runner()
    runner.input_batch.sampling.seed[0] = 42
    assert not _eligible(runner)

    runner = _runner()
    runner.input_batch.repetition_penalties_reqs.add("request-0")
    assert not _eligible(runner)

    runner = _runner()
    runner.input_batch.sampling.top_k[1] = 100
    assert not _eligible(runner)
