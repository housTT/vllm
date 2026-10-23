"""Host-only release contract checks against the isolated vLLM checkout."""

import json
import unittest
from unittest.mock import patch

from vllm.entrypoints.chat_utils import _parse_chat_message_content
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, UsageInfo
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class Tokenizer:
    def get_vocab(self):
        return {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}


class ReasoningContract(unittest.TestCase):
    def test_wire_roundtrip_keeps_internal_reasoning(self):
        for cls, extra in ((DeltaMessage, {}), (ChatMessage, {"role": "assistant"})):
            with self.subTest(cls=cls):
                obj = cls(reasoning="Check", **extra)
                restored = cls.model_validate_json(obj.model_dump_json())
                self.assertEqual(restored.reasoning, "Check")
                self.assertEqual(
                    cls(reasoning_content="Check", **extra).reasoning, "Check"
                )

    def test_nonstream_wire_field(self):
        reason, content = Qwen3ReasoningParser(Tokenizer()).extract_reasoning(
            "Check the result.</think>42", None
        )
        response = ChatCompletionResponse(
            model="ornith-ai/Ornith-1.5-9B",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", reasoning=reason, content=content
                    ),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
        for wire in (response.model_dump(), json.loads(response.model_dump_json())):
            message = wire["choices"][0]["message"]
            self.assertEqual(message.get("reasoning_content"), "Check the result.")
            self.assertNotIn("reasoning", message)
            self.assertEqual(message["content"], "42")

    def test_stream_wire_field(self):
        parser = Qwen3ReasoningParser(Tokenizer())
        delta = parser.extract_reasoning_streaming("", "Check", "Check", [], [5], [5])
        response = ChatCompletionStreamResponse(
            model="ornith-ai/Ornith-1.5-9B",
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=delta)],
        )
        wire = json.loads(response.model_dump_json(exclude_unset=True))
        self.assertEqual(wire["choices"][0]["delta"], {"reasoning_content": "Check"})
        content = DeltaMessage(content="42")
        self.assertEqual(
            json.loads(content.model_dump_json(exclude_unset=True)), {"content": "42"}
        )
        self.assertEqual(delta.reasoning, "Check")

    def test_followup_preserves_reasoning(self):
        for key in ("reasoning_content", "reasoning"):
            with (
                self.subTest(key=key),
                patch(
                    "vllm.entrypoints.chat_utils._parse_chat_message_content_parts",
                    return_value=[{"role": "assistant", "content": "42"}],
                ),
            ):
                request = ChatCompletionRequest(
                    model="ornith-ai/Ornith-1.5-9B",
                    messages=[{"role": "assistant", "content": "42", key: "Check"}],
                )
                result = _parse_chat_message_content(
                    request.messages[0],
                    None,
                    "string",
                    False,
                )
                self.assertEqual(result[0].get("reasoning_content"), "Check")
                self.assertEqual(result[0].get("reasoning"), "Check")


if __name__ == "__main__":
    unittest.main()
