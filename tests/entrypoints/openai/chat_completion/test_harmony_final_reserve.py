# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Harmony (gpt-oss) final-channel reserve in chat completions.

The engine is scripted: each ``generate`` call replays a fixed list of
``RequestOutput``s and records what it was asked for, so the tests can check
the two-phase continuation without a model.
"""

import copy
import json
from contextlib import suppress
from typing import Any
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from tests.entrypoints.openai.chat_completion.test_serving_chat import (
    GPT_OSS_MODEL_NAME,
    MODEL_NAME,
    MockHFConfig,
    MockModelConfig,
    _build_renderer,
    _build_serving_chat,
)
from tests.entrypoints.openai.utils import accumulate_streaming_response
from vllm.entrypoints.openai.chat_completion.harmony_final_reserve import (
    HARMONY_FORCE_FINAL_TOKEN_IDS,
    HarmonyChannelTracker,
    final_reserve_tokens,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.stats import RequestStateStats

RESERVE = 8
MAX_TOKENS = 64
PROMPT_IDS = [7, 7, 7]  # what the scripted engine reports as prompt for phase 1
PHASE2_PROMPT_IDS = [9] * 40  # deliberately different, must never be surfaced
FORCED = list(HARMONY_FORCE_FINAL_TOKEN_IDS)

ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
FINAL_OPEN = "<|start|>assistant<|channel|>final<|message|>"


def encode(harmony_str: str) -> list[int]:
    return get_encoding().encode(harmony_str, allowed_special="all")


def outputs_for(
    token_ids: list[int],
    finish_reason: str,
    *,
    stream: bool,
    prompt_token_ids: list[int],
    metrics: RequestStateStats | None = None,
) -> list[RequestOutput]:
    """Script one engine request: token-by-token deltas or one final output."""

    def make(ids: list[int], finished: bool) -> RequestOutput:
        return RequestOutput(
            request_id="scripted",
            prompt="prompt",
            prompt_token_ids=list(prompt_token_ids),
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=ids,
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=finish_reason if finished else None,
                    stop_reason=None,
                )
            ],
            finished=finished,
            metrics=metrics,
        )

    if not stream:
        return [make(token_ids, True)]
    deltas = [make([t], False) for t in token_ids[:-1]]
    deltas.append(make(token_ids[-1:], True))
    return deltas


class ScriptedEngine:
    """Replays scripted outputs per ``generate`` call and records the calls."""

    def __init__(self, scripts: list[list[RequestOutput] | Exception]):
        self.scripts = scripts
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        index = len(self.calls)
        self.calls.append(
            {
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "max_tokens": sampling_params.max_tokens,
                "thinking_token_budget": sampling_params.thinking_token_budget,
                "request_id": request_id,
                "kwargs": kwargs,
            }
        )
        script = self.scripts[index]
        if isinstance(script, Exception):
            raise script
        for out in script:
            yield out


def build_chat(scripts, *, harmony: bool = True) -> tuple[OpenAIServingChat, Any]:
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = MockModelConfig()
    engine.model_config.max_model_len = 4096
    if harmony:
        engine.model_config.hf_config = MockHFConfig(model_type="gpt_oss")
        engine.model_config.hf_text_config = MockHFConfig(model_type="gpt_oss")
        # The Harmony renderer does not need the HF tokenizer, but the
        # parser's off-grammar fallback decodes with it.
        engine.model_config.model = GPT_OSS_MODEL_NAME
        engine.model_config.tokenizer = GPT_OSS_MODEL_NAME
    engine.input_processor = MagicMock()
    engine.renderer = _build_renderer(engine.model_config)
    scripted = ScriptedEngine(scripts)
    engine.generate = scripted.generate
    if harmony:
        chat = _build_serving_chat(
            engine,
            reasoning_parser="openai_gptoss",
            tool_parser="openai",
            enable_auto_tools=True,
        )
    else:
        chat = _build_serving_chat(engine)
    return chat, scripted


def make_request(**overrides) -> ChatCompletionRequest:
    fields: dict[str, Any] = dict(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
        max_tokens=MAX_TOKENS,
    )
    fields.update(overrides)
    return ChatCompletionRequest(**fields)


def raw_request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.state = MagicMock()
    return req


async def collect_sse(generator) -> list[dict[str, Any]]:
    chunks = []
    async for line in generator:
        payload = line.removeprefix("data: ").strip()
        if not payload or payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


async def complete(chat, req, stream: bool):
    """Run a request; return the response (accumulated for streams)."""
    result = await chat.create_chat_completion(req, raw_request())
    if isinstance(result, ErrorResponse):
        return result
    if stream:
        return await accumulate_streaming_response(result)
    return result


@pytest.fixture(autouse=True)
def fixed_reserve(monkeypatch):
    # The module attribute wins over ``envs.__getattr__``.
    monkeypatch.setattr(
        envs, "VLLM_HARMONY_FINAL_RESERVE_TOKENS", RESERVE, raising=False
    )
    monkeypatch.setattr(envs, "VLLM_HARMONY_FINAL_TRANSITION_TEXT", "", raising=False)


@pytest.fixture(params=[False, True], ids=["non_stream", "stream"])
def stream(request) -> bool:
    return request.param


def test_forced_token_ids_match_encoding():
    assert list(HARMONY_FORCE_FINAL_TOKEN_IDS) == encode("<|end|>" + FINAL_OPEN)


def test_reserve_clamp(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HARMONY_FINAL_RESERVE_TOKENS", 0, raising=False)
    assert final_reserve_tokens(2048) == 256
    assert final_reserve_tokens(4096) == 512
    assert final_reserve_tokens(16384) == 1024
    assert final_reserve_tokens(512) == 128
    monkeypatch.setattr(envs, "VLLM_HARMONY_FINAL_RESERVE_TOKENS", -1, raising=False)
    assert final_reserve_tokens(2048) == 0
    monkeypatch.setattr(envs, "VLLM_HARMONY_FINAL_RESERVE_TOKENS", 300, raising=False)
    assert final_reserve_tokens(2048) == 300


class TestChannelTracker:
    def test_analysis_body_forces_switch(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(encode(ANALYSIS_OPEN + "We need to think"))
        assert tracker.channel == "analysis"
        assert tracker.forced_continuation_ids([1, 2]) == [1, 2, *FORCED]

    def test_final_body_continues_in_place(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(encode(ANALYSIS_OPEN + "Think.<|end|>" + FINAL_OPEN + "The"))
        assert tracker.channel == "final"
        assert tracker.forced_continuation_ids() == []

    def test_end_of_analysis_forces_the_rest_of_the_switch(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(encode(ANALYSIS_OPEN + "Think.<|end|>"))
        assert tracker.forced_continuation_ids([1, 2]) == FORCED[1:]

    def test_final_header_is_completed(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(
            encode(ANALYSIS_OPEN + "Think.<|end|><|start|>assistant<|channel|>")
        )
        assert tracker.forced_continuation_ids() == FORCED[4:]

        tracker = HarmonyChannelTracker()
        tracker.feed(
            encode(ANALYSIS_OPEN + "Think.<|end|><|start|>assistant<|channel|>final")
        )
        assert tracker.forced_continuation_ids() == FORCED[5:]

    def test_other_header_continues_in_place(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(
            encode(
                ANALYSIS_OPEN + "Think.<|end|><|start|>assistant<|channel|>commentary"
            )
        )
        assert tracker.forced_continuation_ids() == []

    def test_tool_call_passes_through(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(
            encode(
                "<|channel|>commentary to=functions.get_weather "
                '<|constrain|>json<|message|>{"location": "Par'
            )
        )
        assert tracker.forced_continuation_ids() is None

    def test_off_grammar_is_unknown(self):
        tracker = HarmonyChannelTracker()
        tracker.feed(encode(ANALYSIS_OPEN + "Think.<|end|>hello"))
        assert tracker.unknown
        assert tracker.channel is None
        assert tracker.forced_continuation_ids() is None
        tracker.feed(encode("more"))  # stays quiet once unknown
        assert tracker.unknown


@pytest.mark.asyncio
async def test_analysis_cut_off_forces_final_channel(stream):
    reasoning = "We need to think"
    p1 = encode(ANALYSIS_OPEN + reasoning)
    answer = "The answer is 2."
    p2 = encode(answer + "<|return|>")
    metrics_1 = RequestStateStats(
        queued_ts=1.0, scheduled_ts=1.5, first_token_ts=2.0, last_token_ts=3.0
    )
    metrics_2 = RequestStateStats(
        queued_ts=3.1, scheduled_ts=3.2, first_token_ts=3.9, last_token_ts=4.5
    )
    chat, engine = build_chat(
        [
            outputs_for(
                p1,
                "length",
                stream=stream,
                prompt_token_ids=PROMPT_IDS,
                metrics=metrics_1,
            ),
            outputs_for(
                p2,
                "stop",
                stream=stream,
                prompt_token_ids=PHASE2_PROMPT_IDS,
                metrics=metrics_2,
            ),
        ]
    )
    req = make_request(
        stream=stream,
        stream_options={"include_usage": True} if stream else None,
    )

    result = await chat.create_chat_completion(req, raw_request())
    if stream:
        chunks = await collect_sse(result)
        finishes = [
            c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")
        ]
        assert finishes[-1] == "stop"
        assert all(f is None for f in finishes[:-1]), (
            "phase 1 must not close the stream"
        )
        usage_chunks = [c["usage"] for c in chunks if c.get("usage")]
        assert usage_chunks, "final usage chunk expected"
        usage = usage_chunks[-1]
        assert usage["prompt_tokens"] == len(PROMPT_IDS)
        assert usage["completion_tokens"] == len(p1) + len(FORCED) + len(p2)
        assert usage["completion_tokens_details"]["reasoning_tokens"] > 0
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        assert "".join(d.get("reasoning") or "" for d in deltas) == reasoning
        assert "".join(d.get("content") or "" for d in deltas) == answer
    else:
        assert isinstance(result, ChatCompletionResponse)
        choice = result.choices[0]
        assert choice.message.reasoning == reasoning
        assert choice.message.content == answer
        assert choice.finish_reason == "stop"
        assert result.usage.prompt_tokens == len(PROMPT_IDS)
        assert result.usage.completion_tokens == len(p1) + len(FORCED) + len(p2)
        assert result.usage.completion_tokens_details is not None
        assert 0 < result.usage.completion_tokens_details.reasoning_tokens <= len(p1)

    assert len(engine.calls) == 2
    first, second = engine.calls
    assert first["max_tokens"] == MAX_TOKENS - RESERVE
    assert first["thinking_token_budget"] is None
    prompt = first["prompt_token_ids"]
    assert second["prompt_token_ids"] == prompt + p1 + FORCED
    assert second["max_tokens"] == MAX_TOKENS - len(p1) - len(FORCED)
    assert second["request_id"] == first["request_id"] + "-final"
    assert second["kwargs"]["reasoning_ended"] is True
    assert second["thinking_token_budget"] is None


@pytest.mark.asyncio
async def test_final_cut_off_continues_in_place(stream):
    p1 = encode(ANALYSIS_OPEN + "Think.<|end|>" + FINAL_OPEN + "The answer")
    p2 = encode(" is 2.<|return|>")
    chat, engine = build_chat(
        [
            outputs_for(p1, "length", stream=stream, prompt_token_ids=PROMPT_IDS),
            outputs_for(p2, "stop", stream=stream, prompt_token_ids=PHASE2_PROMPT_IDS),
        ]
    )
    response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.reasoning == "Think."
    assert response.choices[0].message.content == "The answer is 2."
    assert response.choices[0].finish_reason == "stop"
    assert len(engine.calls) == 2
    assert (
        engine.calls[1]["prompt_token_ids"] == engine.calls[0]["prompt_token_ids"] + p1
    )
    assert engine.calls[1]["max_tokens"] == MAX_TOKENS - len(p1)


@pytest.mark.asyncio
async def test_normal_stop_passes_through(stream):
    p1 = encode(ANALYSIS_OPEN + "Think.<|end|>" + FINAL_OPEN + "Two.<|return|>")
    chat, engine = build_chat(
        [outputs_for(p1, "stop", stream=stream, prompt_token_ids=PROMPT_IDS)]
    )
    response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.content == "Two."
    assert response.choices[0].finish_reason == "stop"
    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == MAX_TOKENS - RESERVE
    if not stream:
        details = response.usage.completion_tokens_details
        assert details is not None and details.reasoning_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    [
        "<|end|>",
        "<|end|><|start|>",
        "<|end|><|start|>assistant<|channel|>",
        "<|end|><|start|>assistant<|channel|>final",
    ],
    ids=["after_end", "after_start", "in_channel", "after_final"],
)
async def test_header_cut_off_completes_the_switch(stream, header):
    p1 = encode(ANALYSIS_OPEN + "Think." + header)
    p2 = encode("The answer is 2.<|return|>")
    chat, engine = build_chat(
        [
            outputs_for(p1, "length", stream=stream, prompt_token_ids=PROMPT_IDS),
            outputs_for(p2, "stop", stream=stream, prompt_token_ids=PHASE2_PROMPT_IDS),
        ]
    )
    response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.reasoning == "Think."
    assert response.choices[0].message.content == "The answer is 2."
    assert response.choices[0].finish_reason == "stop"
    assert len(engine.calls) == 2
    rest = FORCED[len(encode(header)) :]
    assert (
        engine.calls[1]["prompt_token_ids"]
        == engine.calls[0]["prompt_token_ids"] + p1 + rest
    )
    assert engine.calls[1]["max_tokens"] == MAX_TOKENS - len(p1) - len(rest)
    assert engine.calls[1]["kwargs"]["reasoning_ended"] is True


@pytest.mark.asyncio
async def test_other_header_cut_off_continues_in_place(stream):
    p1 = encode(
        ANALYSIS_OPEN + "Think.<|end|><|start|>assistant<|channel|>commentary"
    )
    p2 = encode("<|message|>Two.<|return|>")
    chat, engine = build_chat(
        [
            outputs_for(p1, "length", stream=stream, prompt_token_ids=PROMPT_IDS),
            outputs_for(p2, "stop", stream=stream, prompt_token_ids=PHASE2_PROMPT_IDS),
        ]
    )
    response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.reasoning == "Think."
    assert response.choices[0].finish_reason == "stop"
    assert len(engine.calls) == 2
    assert (
        engine.calls[1]["prompt_token_ids"] == engine.calls[0]["prompt_token_ids"] + p1
    )
    assert engine.calls[1]["max_tokens"] == MAX_TOKENS - len(p1)
    assert engine.calls[1]["kwargs"]["reasoning_ended"] is False


@pytest.mark.asyncio
async def test_off_grammar_output_passes_through(stream):
    p1 = encode(ANALYSIS_OPEN + "Think.<|end|>hello")
    chat, engine = build_chat(
        [outputs_for(p1, "length", stream=stream, prompt_token_ids=PROMPT_IDS)]
    )
    response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.reasoning == "Think."
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "length"
    assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_reserve_smaller_than_forced_sequence_passes_through(monkeypatch):
    monkeypatch.setattr(
        envs,
        "VLLM_HARMONY_FINAL_TRANSITION_TEXT",
        "I must stop here and give the answer now based on what I have.",
        raising=False,
    )
    p1 = encode(ANALYSIS_OPEN + " ".join(["think"] * (MAX_TOKENS - RESERVE - 3)))
    chat, engine = build_chat(
        [outputs_for(p1, "length", stream=False, prompt_token_ids=PROMPT_IDS)]
    )
    response = await complete(chat, make_request(), stream=False)
    assert response.choices[0].message.content is None
    assert response.choices[0].finish_reason == "length"
    assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_transition_text_is_forced_into_analysis():
    reasoning = "We need to think"
    transition = " I must answer now."
    p1 = encode(ANALYSIS_OPEN + reasoning)
    p2 = encode("Two.<|return|>")
    chat, engine = build_chat(
        [
            outputs_for(p1, "length", stream=False, prompt_token_ids=PROMPT_IDS),
            outputs_for(p2, "stop", stream=False, prompt_token_ids=PHASE2_PROMPT_IDS),
        ]
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            envs, "VLLM_HARMONY_FINAL_TRANSITION_TEXT", transition, raising=False
        )
        response = await complete(chat, make_request(), stream=False)
    transition_ids = chat.renderer.tokenizer.encode(
        transition, add_special_tokens=False
    )
    assert engine.calls[1]["prompt_token_ids"] == (
        engine.calls[0]["prompt_token_ids"] + p1 + transition_ids + FORCED
    )
    assert response.choices[0].message.reasoning == reasoning + transition
    assert response.choices[0].message.content == "Two."


@pytest.mark.asyncio
async def test_thinking_token_budget_caps_phase_one():
    p1 = encode(ANALYSIS_OPEN + "Think.<|end|>" + FINAL_OPEN + "Two.<|return|>")
    chat, engine = build_chat(
        [outputs_for(p1, "stop", stream=False, prompt_token_ids=PROMPT_IDS)]
    )
    response = await complete(chat, make_request(thinking_token_budget=5), stream=False)
    assert isinstance(response, ChatCompletionResponse)
    assert engine.calls[0]["max_tokens"] == 5
    assert engine.calls[0]["thinking_token_budget"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 2},
        {"logprobs": True, "top_logprobs": 1},
        {"ignore_eos": True},
        {"max_tokens": 2 * RESERVE - 1},
        {
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
        },
    ],
    ids=["n2", "logprobs", "ignore_eos", "small_budget", "required_tool"],
)
async def test_bypass_conditions_use_the_full_budget(overrides):
    p1 = encode(ANALYSIS_OPEN + "Think")
    chat, engine = build_chat(
        [outputs_for(p1, "length", stream=False, prompt_token_ids=PROMPT_IDS)]
    )
    req = make_request(**overrides)
    with suppress(Exception):
        await complete(chat, req, stream=False)
    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == req.max_tokens

    # A budget request that hits a bypass is refused rather than ignored.
    chat, engine = build_chat(
        [outputs_for(p1, "length", stream=False, prompt_token_ids=PROMPT_IDS)]
    )
    result = await complete(
        chat, make_request(thinking_token_budget=4, **overrides), False
    )
    assert isinstance(result, ErrorResponse)
    assert engine.calls == []


@pytest.mark.asyncio
async def test_disabled_reserve_passes_through(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HARMONY_FINAL_RESERVE_TOKENS", -1, raising=False)
    p1 = encode(ANALYSIS_OPEN + "Think")
    chat, engine = build_chat(
        [outputs_for(p1, "length", stream=False, prompt_token_ids=PROMPT_IDS)]
    )
    response = await complete(chat, make_request(), stream=False)
    assert response.choices[0].message.content is None
    assert response.choices[0].finish_reason == "length"
    assert engine.calls[0]["max_tokens"] == MAX_TOKENS


@pytest.mark.asyncio
async def test_rejected_continuation_returns_truncated_output(stream, caplog):
    p1 = encode(ANALYSIS_OPEN + "We need to think")
    chat, engine = build_chat(
        [
            outputs_for(p1, "length", stream=stream, prompt_token_ids=PROMPT_IDS),
            ValueError("prompt too long"),
        ]
    )
    with caplog.at_level("WARNING"):
        response = await complete(chat, make_request(stream=stream), stream)
    assert response.choices[0].message.reasoning == "We need to think"
    assert response.choices[0].message.content is None
    assert response.choices[0].finish_reason == "length"
    assert len(engine.calls) == 2
    assert any("was not accepted" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_non_harmony_model_is_untouched():
    out = RequestOutput(
        request_id="scripted",
        prompt="prompt",
        prompt_token_ids=PROMPT_IDS,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="plain answer",
                token_ids=[4, 5, 6],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="length",
                stop_reason=None,
            )
        ],
        finished=True,
    )
    chat, engine = build_chat([[copy.copy(out)]], harmony=False)
    response = await complete(chat, make_request(), stream=False)
    assert response.choices[0].message.content == "plain answer"
    assert response.usage.completion_tokens_details is None
    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == MAX_TOKENS
