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

        # Gemma 4 may emit more than one thought block before its answer or
        # tool call, e.g. a closed thought followed by an empty one. The base
        # parser consumes a single start/end pair, so a later complete block
        # would survive into ``content`` and leak the channel markers to the
        # API. Consume every further complete block that leads the remaining
        # content, keeping its reasoning, and hand back the rest byte for byte.
        parts = [reasoning] if reasoning else []
        while content is not None:
            block, rest = _split_leading_thought_block(content, self.start_token, self.end_token)
            if block is None:
                break
            block = _strip_thought_label(block)
            if block:
                parts.append(block)
            content = rest or None

        if parts:
            reasoning = "\n".join(parts)
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


def _split_leading_thought_block(
    text: str, start_token: str, end_token: str
) -> tuple[str | None, str]:
    """Split one complete thought block off the front of ``text``.

    Only whitespace may precede the block, and it must be closed; anything else
    is left untouched so ordinary content or a quoted tool argument that merely
    resembles a marker is never removed. Returns ``(block_body, remainder)`` or
    ``(None, text)`` when nothing was consumed.
    """
    stripped = text.lstrip()
    if not stripped.startswith(start_token):
        return None, text
    body_start = len(start_token)
    end_index = stripped.find(end_token, body_start)
    if end_index == -1:
        return None, text
    return stripped[body_start:end_index], stripped[end_index + len(end_token) :]
