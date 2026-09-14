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


# Wire form of the bash tool call from the retained SWE response
# chatcmpl-b685fcadaa7cbafe; only the command text is shortened.
_TOOL = '<|tool_call>call:bash{command:<|"|>python reproduce_issue.py<|"|>}<tool_call|>'
_EMPTY_BLOCK = "<|channel>thought\n<channel|>"


def test_repeated_empty_thought_blocks_do_not_leak_into_content():
    # Captured 2026-09-11: the API returned content exactly equal to
    # _EMPTY_BLOCK because only the first block was consumed.
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = parser.extract_reasoning(_EMPTY_BLOCK + _EMPTY_BLOCK + _TOOL, request=None)

    assert reasoning == ""
    assert content == _TOOL


def test_retained_leaked_content_alone_yields_no_content():
    # Post-parser replay of the exact retained HTTP content.
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = parser.extract_reasoning(_EMPTY_BLOCK, request=None)

    assert reasoning == ""
    assert content is None


def test_later_thought_block_reasoning_is_kept():
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = parser.extract_reasoning(
        _EMPTY_BLOCK + "<|channel>thought\nrun the script<channel|>" + _TOOL, request=None
    )

    assert reasoning == "run the script"
    assert content == _TOOL


def test_two_nonempty_thought_blocks_are_joined():
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = parser.extract_reasoning(
        "<|channel>thought\nfirst<channel|>\n<|channel>thought\nsecond<channel|>done", request=None
    )

    assert reasoning == "first\nsecond"
    assert content == "done"


def test_incomplete_later_block_is_left_in_content():
    parser = Gemma4ReasoningParser(_Tokenizer())
    tail = "<|channel>thought\nstill thinking"

    reasoning, content = parser.extract_reasoning(_EMPTY_BLOCK + tail, request=None)

    assert reasoning == ""
    assert content == tail


def test_text_before_later_block_is_not_consumed():
    parser = Gemma4ReasoningParser(_Tokenizer())
    tail = "answer" + _EMPTY_BLOCK

    reasoning, content = parser.extract_reasoning(_EMPTY_BLOCK + tail, request=None)

    assert reasoning == ""
    assert content == tail


def test_content_ids_start_after_last_closed_block():
    parser = Gemma4ReasoningParser(_Tokenizer())

    assert parser.extract_content_ids([1, 9, 2, 1, 9, 2, 4, 7]) == [4, 7]
    assert parser.extract_content_ids([1, 9, 2, 1, 9]) == []
    assert parser.extract_content_ids([1, 9]) == []


def _stream(parser, deltas):
    """Feed (text, ids) deltas; return collected reasoning and content."""
    reasoning, content = [], []
    prev_text, prev_ids = "", []
    for text, ids in deltas:
        cur_text, cur_ids = prev_text + text, prev_ids + ids
        result = parser.extract_reasoning_streaming(prev_text, cur_text, text, prev_ids, cur_ids, ids)
        prev_text, prev_ids = cur_text, cur_ids
        if result is None:
            continue
        if result.reasoning:
            reasoning.append(result.reasoning)
        if result.content:
            content.append(result.content)
    return "".join(reasoning), "".join(content)


def test_streaming_reopened_thought_block_is_reasoning_not_content():
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = _stream(
        parser,
        [
            ("<|channel>", [1]),
            ("thought\n", [10]),
            ("plan", [11]),
            ("<channel|>", [2]),
            ("<|channel>", [1]),
            ("thought\n", [10]),
            ("<channel|>", [2]),
            ("<|tool_call>", [4]),
            ("call:bash{}", [12]),
            ("<tool_call|>", [13]),
        ],
    )

    assert reasoning == "plan"
    assert content == "<|tool_call>call:bash{}<tool_call|>"


def test_streaming_single_block_unchanged():
    parser = Gemma4ReasoningParser(_Tokenizer())

    reasoning, content = _stream(
        parser,
        [("<|channel>", [1]), ("thought\n", [10]), ("check", [11]), ("<channel|>", [2]), ("The answer is 4.", [12])],
    )

    assert reasoning == "check"
    assert content == "The answer is 4."
