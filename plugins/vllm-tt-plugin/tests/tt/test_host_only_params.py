# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for sampling parameters which require host ("compat") sampling.
"""

import string

import regex as re

from tests.tt.utils import RequestConfig, run_concurrent_batch


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
        """Test bad_words parameter prevents specified words from appearing."""
        bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
        configs = [
            # Run multiple times with high temperature to increase coverage
            RequestConfig(
                prompt="Say hello to me",
                max_tokens=100,
                bad_words=bad_words,
                temperature=1.0,
                seed=i,
            )
            for i in range(5)
        ]
        # bad_words is only available in chat completions API
        results = run_concurrent_batch(tt_server, tt_model_name, configs, use_chat=True)
        assert len(results) == len(configs)

        for i, text in enumerate(results):
            assert text is not None, f"Response {i} content is None"
            # Strip punctuation except '>' to avoid false positives from
            # BPE-merged tokens like ">Hello" (a single token distinct from "Hello")
            punct = string.punctuation.replace(">", "")
            words = [word.strip(punct) for word in text.split()]

            # Check if any bad word appears as a whole word (case-insensitive)
            for bad_word in bad_words:
                assert bad_word not in words, (
                    f"bad_word '{bad_word}' found as whole word"
                    f"in response {i}: {text!r}"
                )

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
        """Test allowed_token_ids without assuming low IDs decode visibly."""
        allowlists = ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])
        configs = [
            RequestConfig(
                prompt="Allowed: ",
                max_tokens=10,
                allowed_token_ids=list(allowed),
                logprobs=0,
                return_tokens_as_token_ids=True,
            )
            for allowed in allowlists
        ]
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, return_full_response=True
        )
        assert len(results) == len(configs)
        # Inspect IDs rather than decoded text: low IDs can be EOS/control
        # tokens and legitimately decode to an empty string for some models.
        for i, (response, allowed) in enumerate(zip(results, allowlists)):
            assert response.usage.completion_tokens > 0, (
                f"should produce at least one token for request {i}"
            )
            tokens = response.choices[0].logprobs.tokens
            token_ids = [
                int(value)
                for token in tokens
                for value in re.findall(r"token_id:(\d+)", token)
            ]
            assert token_ids, f"should produce at least one token for request {i}"
            assert set(token_ids) <= set(allowed), (
                f"request {i} produced IDs outside {allowed}: {token_ids}"
            )

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
