# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for sampling parameters which require host ("compat") sampling.
"""

from tests.tt.utils import RequestConfig, run_concurrent_batch
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer


def _assert_no_bad_word_sequences(token_ids, bad_word_sequences):
    for sequence in bad_word_sequences:
        for start in range(len(token_ids) - len(sequence) + 1):
            assert token_ids[start : start + len(sequence)] != sequence, (
                f"Banned token sequence {sequence} starts at output index {start}: "
                f"{token_ids}"
            )


class TestHostOnlyParameters:
    def test_min_p(self, tt_server, tt_model_name, max_batch_size):
        """Test min_p parameter (smoke test - verifies it doesn't error)."""
        configs = [
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.1),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.2),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.3),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.4),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.5),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.6),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.7),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.8),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.9),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=1.0),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # min_p affects sampling distribution - just verify we get output
        assert results[0] is not None and len(results[0]) > 0, (
            "should produce non-empty output"
        )

    def test_bad_words(self, tt_server, tt_model_name, max_batch_size):
        """Test bad_words excludes the tokenizer's forbidden token sequences."""
        bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
        sampling_params = SamplingParams(bad_words=bad_words)
        sampling_params.update_from_tokenizer(get_tokenizer(tt_model_name))
        bad_word_sequences = sampling_params.bad_words_token_ids
        assert bad_word_sequences
        configs = [
            # Run multiple times with high temperature to increase coverage
            RequestConfig(
                prompt="Say hello to me",
                max_tokens=100,
                bad_words=bad_words,
                temperature=1.0,
                seed=i,
                return_token_ids=True,
            )
            for i in range(5)
        ]
        # bad_words is only available in chat completions API
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, use_chat=True, return_full_response=True
        )
        assert len(results) == len(configs)

        for i, response in enumerate(results):
            choice = response.choices[0]
            assert choice.message.content, f"Response {i} content is empty"
            token_ids = choice.model_dump().get("token_ids")
            assert token_ids, f"Response {i} generated token IDs are missing"
            # vLLM bans token sequences, not every spelling-equivalent BPE
            # segmentation. For example, allowed tokens "h" + "ello" must not
            # be confused with the banned singleton "hello" by re-tokenizing.
            _assert_no_bad_word_sequences(token_ids, bad_word_sequences)

    def test_logit_bias(self, tt_server, tt_model_name, max_batch_size):
        """Test logit_bias parameter (smoke test)."""
        configs = [
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={1: 0.1}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={4: 0.2}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={2: 0.3}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={16: 0.4}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={7: 0.5}),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        assert results[0] is not None, "should produce output with logit_bias"

    def test_allowed_token_ids(self, tt_server, tt_model_name, max_batch_size):
        """Test allowed_token_ids parameter (smoke test)."""
        configs = [
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[1, 2, 3]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[4, 5, 6]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[7, 8, 9]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[10, 11, 12]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[13, 14, 15]
            ),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # With only 3 allowed tokens, output should be limited
        for i, result in enumerate(results):
            assert result is not None, f"should produce output for request {i}"
            assert len(result) > 0, f"should produce non-empty output for request {i}"

    def test_min_tokens(self, tt_server, tt_model_name, max_batch_size):
        """Test min_tokens parameter ensures minimum output length."""
        min_tokens = 5
        configs = [
            # Use a prompt that might naturally produce short output
            RequestConfig(prompt="Say OK.", max_tokens=100, min_tokens=min_tokens),
        ]
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, return_full_response=True
        )
        assert len(results) == len(configs)

        response = results[0]
        # Check usage stats for token count
        assert response.usage is not None, "usage stats should be returned"
        assert response.usage.completion_tokens >= min_tokens, (
            f"should produce at least {min_tokens} tokens, "
            f"got {response.usage.completion_tokens}"
        )
