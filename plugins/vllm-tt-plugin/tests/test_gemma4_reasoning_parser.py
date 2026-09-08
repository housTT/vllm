# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser


class _Tokenizer:
    def get_vocab(self):
        return {
            "<|channel>": 1,
            "<channel|>": 2,
            "<|turn>": 3,
            "<|tool_call>": 4,
            "<|tool_response>": 5,
        }


def test_extract_reasoning_strips_gemma_thought_label():
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = parser.extract_reasoning(
        "<|channel>thought\ncheck the arithmetic<channel|>The answer is 4.",
        request=None,
    )

    assert reasoning == "check the arithmetic"
    assert content == "The answer is 4."


def test_adjust_request_preserves_special_tokens_for_parser():
    parser = Gemma4ReasoningParser(_Tokenizer())
    request = SimpleNamespace(skip_special_tokens=True)

    assert parser.adjust_request(request) is request
    assert request.skip_special_tokens is False


def test_tool_call_terminates_reasoning_channel():
    parser = Gemma4ReasoningParser(_Tokenizer())

    assert parser.is_reasoning_end([1, 4]) is True
    assert parser.is_reasoning_end([2, 3, 1]) is False
