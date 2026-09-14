# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_THOUGHT_PREFIX = "thought\n"
_FENCED_THOUGHT_PREFIX = "```thought\n"
_TOOL_CALL_START = "<|tool_call>"


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Google Gemma4 unified thinking models."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content starts after the *last* closed thought block. A block that is
        # re-opened and not yet closed means there is no content yet.
        last_end = _rindex(input_ids, self.end_token_id)
        if last_end is None:
            return []
        last_start = _rindex(input_ids, self.start_token_id)
        if last_start is not None and last_start > last_end:
            return []
        return input_ids[last_end + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)

        # Gemma 4 emits more than one thought block per turn: a closed thought
        # followed by an empty one, or visible text followed by an empty block
        # right before the tool call. The base parser consumes a single
        # start/end pair, so every further block would reach the API as
        # content. Consume each complete block in the text that precedes the
        # first tool call, keep its reasoning, and leave the tool-call wire
        # string byte for byte, so a quoted argument is never touched.
        if content is not None:
            head, sep, tail = content.partition(_TOOL_CALL_START)
            extra, consumed_head = _consume_thought_blocks(head, self.start_token, self.end_token)
            if consumed_head != head:
                head = consumed_head.lstrip()
            parts = [reasoning] if reasoning else []
            parts.extend(extra)
            if parts:
                reasoning = "\n".join(parts)
            content = (head + sep + tail) or None
        return reasoning, content

    def _reasoning_open(self, token_ids: Sequence[int]) -> bool:
        """True when the most recent channel marker in ``token_ids`` is a start."""
        for token_id in reversed(token_ids):
            if token_id == self.start_token_id:
                return True
            if token_id == self.end_token_id:
                return False
        return False

    def _stream_delta(
        self,
        delta_text: str,
        previous_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Classify one streaming delta, allowing a thought block to re-open.

        The base implementation decides from whether the start/end tokens
        appear *anywhere* in the previous ids, so once a block has closed every
        later delta is content. Gemma 4 can open another block after that; its
        label and text would then stream out as content. Decide from the most
        recent marker instead.
        """
        start_id, end_id = self.start_token_id, self.end_token_id
        if len(delta_token_ids) == 1 and delta_token_ids[0] == start_id:
            # A block opens: reset the per-block label state before its label
            # arrives in the next delta.
            self._reasoning_text = ""
            self._prefix_stripped = False
            return None
        if len(delta_token_ids) == 1 and delta_token_ids[0] == end_id:
            return None

        if self._reasoning_open(previous_token_ids):
            if end_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=reasoning, content=content or None)
            return DeltaMessage(reasoning=delta_text)

        if start_id in delta_token_ids:
            # A block opens in this delta: reset the per-block label state.
            self._reasoning_text = ""
            self._prefix_stripped = False
            start_index = delta_text.find(self.start_token)
            body = delta_text[start_index + len(self.start_token) :]
            if end_id in delta_token_ids:
                end_index = body.find(self.end_token)
                reasoning = body[:end_index]
                content = body[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=reasoning, content=content or None)
            return DeltaMessage(reasoning=body)

        return DeltaMessage(content=delta_text)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        result = self._stream_delta(delta_text, previous_token_ids, delta_token_ids)
        if result is None:
            return None

        if result.reasoning is None:
            return result

        self._reasoning_text += result.reasoning

        if self._prefix_stripped:
            return result

        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                self._prefix_stripped = True
                return result

            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = result.reasoning[chars_of_prefix_in_delta:]
            if stripped:
                self._prefix_stripped = True
                result.reasoning = stripped
                return result

            if len(self._reasoning_text) >= prefix_len:
                self._prefix_stripped = True
                result.reasoning = ""
                return result
            return None

        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result


def _strip_thought_label(text: str) -> str:
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _rindex(values: Sequence[int], target: int) -> int | None:
    for index in range(len(values) - 1, -1, -1):
        if values[index] == target:
            return index
    return None


def _consume_thought_blocks(text: str, start_token: str, end_token: str) -> tuple[list[str], str]:
    """Remove every complete thought block from ``text``.

    Returns ``(reasoning_parts, remaining_text)``. A block that is opened but
    never closed is left in place (the model may still be generating it). A
    stray ``end_token`` with no opener closes a thought the model began without
    the start token (seen as a markdown ``thought`` fence): the text before it
    is reasoning. Empty blocks contribute no reasoning.
    """
    parts: list[str] = []
    remaining: list[str] = []
    cursor = 0
    while True:
        start = text.find(start_token, cursor)
        end = text.find(end_token, cursor)
        if end != -1 and (start == -1 or end < start):
            # stray closer: everything since the cursor was thought text
            body = _strip_thought_label(text[cursor:end].lstrip())
            body = _strip_fenced_thought_label(body)
            if body.strip():
                parts.append(body.rstrip())
            cursor = end + len(end_token)
            continue
        if start == -1:
            break
        close = text.find(end_token, start + len(start_token))
        if close == -1:
            break  # incomplete block: leave it
        remaining.append(text[cursor:start])
        body = _strip_thought_label(text[start + len(start_token) : close])
        if body.strip():
            parts.append(body.rstrip())
        cursor = close + len(end_token)
    remaining.append(text[cursor:])
    return parts, "".join(remaining)


def _strip_fenced_thought_label(text: str) -> str:
    if text.startswith(_FENCED_THOUGHT_PREFIX):
        return text[len(_FENCED_THOUGHT_PREFIX) :]
    return text
