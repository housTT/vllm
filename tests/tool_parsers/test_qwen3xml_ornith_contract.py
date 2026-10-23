# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only qwen3_xml contracts for fragmented typed tool arguments."""

import json
import unittest
from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser

EXPECTED = {
    "text": 'Paris & Lyon, "été"\nline two',
    "count": 3,
    "enabled": True,
    "ratio": 0.25,
    "items": ["a", "b"],
    "metadata": {"key": "value", "n": 2},
}
TOOL = {
    "type": "function",
    "function": {
        "name": "release_probe",
        "description": "Return structured probe data.",
        "parameters": {
            "type": "object",
            "properties": {
                name: {"type": kind}
                for name, kind in (
                    ("text", "string"),
                    ("count", "integer"),
                    ("enabled", "boolean"),
                    ("ratio", "number"),
                    ("items", "array"),
                    ("metadata", "object"),
                )
            },
            "required": list(EXPECTED),
        },
    },
}


def xml_call(values):
    parts = ["<tool_call>\n<function=release_probe>\n"]
    for key, value in values.items():
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        parts.append(f"<parameter={key}>{text}</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)


class ToolParserContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # XML parsing consumes decoded strings; tokenizer operations are unused.
        cls.tokenizer = SimpleNamespace(get_vocab=lambda: {})
        cls.request = ChatCompletionRequest(
            model="ornith-ai/Ornith-1.5-9B", messages=[], tools=[TOOL]
        )

    def assert_calls(self, calls, expected):
        self.assertEqual(len(calls), len(expected))
        ids = []
        for call, arguments in zip(calls, expected):
            self.assertEqual(call["function"]["name"], "release_probe")
            self.assertEqual(json.loads(call["function"]["arguments"]), arguments)
            self.assertTrue(call["id"])
            ids.append(call["id"])
        self.assertEqual(len(ids), len(set(ids)))

    def parse_stream(self, parser, chunks):
        current = ""
        calls = {}
        for chunk in chunks:
            previous = current
            current += chunk
            delta = parser.extract_tool_calls_streaming(
                previous, current, chunk, [], [], [], self.request
            )
            for call in delta.tool_calls or []:
                found = calls.setdefault(
                    call.index, {"id": "", "function": {"name": "", "arguments": ""}}
                )
                if call.id:
                    # The parser may repeat a stable call id; it must never change.
                    self.assertIn(found["id"], ("", call.id))
                    found["id"] = call.id
                if call.function:
                    for field in ("name", "arguments"):
                        found["function"][field] += getattr(call.function, field) or ""
        return [calls[index] for index in sorted(calls)]

    def test_nonstream_typed_arguments_and_request_reuse(self):
        parser = Qwen3XMLToolParser(self.tokenizer)
        for values in (EXPECTED, {"text": "second request"}, EXPECTED):
            result = parser.extract_tool_calls(xml_call(values), self.request)
            self.assertTrue(result.tools_called)
            self.assert_calls(
                [call.model_dump() for call in result.tool_calls], [values]
            )

    def test_fragmented_stream_typed_arguments(self):
        text = xml_call(EXPECTED)
        for size in (1, 3, 7, 64, len(text)):
            with self.subTest(chunk_characters=size):
                parser = Qwen3XMLToolParser(self.tokenizer)
                chunks = [
                    text[offset : offset + size] for offset in range(0, len(text), size)
                ]
                self.assert_calls(self.parse_stream(parser, chunks), [EXPECTED])

    def test_multiple_tool_calls_streaming_and_nonstreaming(self):
        expected = [EXPECTED, {"text": "second call", "count": 0, "enabled": False}]
        text = "\n".join(xml_call(values) for values in expected)
        parser = Qwen3XMLToolParser(self.tokenizer)
        self.assert_calls(
            [
                call.model_dump()
                for call in parser.extract_tool_calls(text, self.request).tool_calls
            ],
            expected,
        )
        for size in (1, 13, len(text)):
            with self.subTest(chunk_characters=size):
                chunks = [
                    text[offset : offset + size] for offset in range(0, len(text), size)
                ]
                self.assert_calls(self.parse_stream(parser, chunks), expected)

    def test_scalar_values_do_not_change_with_fragmentation(self):
        cases = [
            (
                {"text": "null", "count": -42, "enabled": False, "ratio": "1e-5"},
                {"text": "null", "count": -42, "enabled": False, "ratio": 1e-5},
            ),
            (
                {"text": "", "count": 0, "enabled": True, "ratio": None},
                {"text": "", "count": 0, "enabled": True, "ratio": None},
            ),
        ]
        for raw, expected in cases:
            text = xml_call(raw)
            for size in (1, 5, len(text)):
                with self.subTest(raw=raw, chunk_characters=size):
                    parser = Qwen3XMLToolParser(self.tokenizer)
                    chunks = [
                        text[offset : offset + size]
                        for offset in range(0, len(text), size)
                    ]
                    self.assert_calls(self.parse_stream(parser, chunks), [expected])


if __name__ == "__main__":
    unittest.main(verbosity=2)
