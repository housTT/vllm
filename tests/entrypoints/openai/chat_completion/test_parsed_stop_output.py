# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _truncate_parsed_output_at_stop,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("reasoning", "content", "tool_calls", "include_stop", "expected"),
    [
        (
            "analysis prefixStop",
            None,
            None,
            False,
            ("analysis prefix", None, None),
        ),
        (
            "Stop earlier",
            "answerStop",
            None,
            False,
            ("Stop earlier", "answer", None),
        ),
        (
            "analysis prefixStop-same-token-suffix",
            None,
            None,
            True,
            ("analysis prefixStop", None, None),
        ),
        (
            "analysis",
            "commentary",
            [FunctionCall(name="tool", arguments='{"value":"prefixStop')],
            False,
            ("analysis", "commentary", '{"value":"prefix'),
        ),
    ],
    ids=[
        "reasoning_partial_token",
        "terminal_content",
        "include_stop",
        "terminal_tool_call",
    ],
)
def test_truncate_parsed_output_at_stop(
    reasoning, content, tool_calls, include_stop, expected
):
    result_reasoning, result_content, result_tool_calls = (
        _truncate_parsed_output_at_stop(
            reasoning,
            content,
            tool_calls,
            "Stop",
            include_stop,
        )
    )
    result_tool_arguments = (
        result_tool_calls[-1].arguments if result_tool_calls else None
    )
    assert (result_reasoning, result_content, result_tool_arguments) == expected


def test_truncate_parsed_output_at_stop_ignores_unmatched_text():
    result = _truncate_parsed_output_at_stop(
        "analysis", "answer", None, "Stop", False
    )
    assert result == ("analysis", "answer", None)
