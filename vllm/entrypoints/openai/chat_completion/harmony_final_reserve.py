# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-channel reserve for Harmony (gpt-oss) chat completions.

gpt-oss writes its reasoning in the Harmony ``analysis`` channel and the answer
in the ``final`` channel. When ``max_tokens`` runs out while the analysis
channel is still open, the completion carries ``reasoning`` but ``content`` is
null. The helpers in this module let the chat serving layer keep part of the
budget for the final channel:

1. The request first runs with ``max_tokens - reserve``.
2. If it stops on ``length`` inside the analysis channel, the server forces
   the channel switch (``<|end|><|start|>assistant<|channel|>final<|message|>``)
   and continues in a follow-up engine request whose prompt is the original
   prompt plus the reasoning tokens plus the switch tokens, with the remaining
   budget as ``max_tokens``. If it stops on ``length`` while already writing
   the answer, the follow-up continues without forced tokens.
3. The follow-up outputs are re-based (original prompt fields, cumulative
   token ids) so the rest of the serving code sees one continuous request.

The total number of completion tokens never exceeds the request's
``max_tokens``. The engine does not sample differently; this is purely a
serving-layer split of the budget, which is what lets it work on backends that
sample on device and cannot run per-request logits processors.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from openai_harmony import HarmonyError, StreamState

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_streamable_parser_for_assistant,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.metrics.stats import RequestStateStats

# <|end|> <|start|> assistant <|channel|> final <|message|>
HARMONY_FORCE_FINAL_TOKEN_IDS: Final[tuple[int, ...]] = (
    200007,
    200006,
    173781,
    200005,
    17196,
    200008,
)

MIN_AUTO_RESERVE_TOKENS: Final = 128
MAX_AUTO_RESERVE_TOKENS: Final = 1024


def final_reserve_tokens(max_tokens: int) -> int:
    """Tokens of ``max_tokens`` kept for the final channel; 0 when disabled."""
    setting = envs.VLLM_HARMONY_FINAL_RESERVE_TOKENS
    if setting < 0:
        return 0
    if setting > 0:
        return setting
    return max(MIN_AUTO_RESERVE_TOKENS, min(MAX_AUTO_RESERVE_TOKENS, max_tokens // 8))


@dataclass
class FinalReservePlan:
    max_tokens: int
    """The request's budget, as bounded by the serving layer."""
    phase1_max_tokens: int
    """Budget of the first engine request."""
    reserve: int
    """Tokens kept back for the final channel."""
    transition_ids: list[int]
    """Optional text forced into the analysis message before the switch."""


def plan_final_reserve(
    request: ChatCompletionRequest,
    sampling_params: SamplingParams,
    thinking_token_budget: int | None,
    tokenizer: TokenizerLike,
) -> FinalReservePlan | None:
    """Decide whether and how to split the budget.

    Returns ``None`` when the request must pass through unchanged: the
    feature is disabled, the budget is too small to split, or the request uses
    a feature the two-phase continuation cannot honour (several sequences,
    logprobs, ``ignore_eos``, a mandated tool call).
    """
    max_tokens = sampling_params.max_tokens
    if max_tokens is None:
        return None
    reserve = final_reserve_tokens(max_tokens)
    if reserve <= 0 or max_tokens < 2 * reserve:
        return None
    if sampling_params.n != 1:
        return None
    if (
        sampling_params.logprobs is not None
        or sampling_params.logprob_token_ids
        or sampling_params.prompt_logprobs is not None
    ):
        return None
    if sampling_params.ignore_eos:
        return None
    if sampling_params.output_kind not in (
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ):
        return None
    if request.tool_choice == "required" or isinstance(
        request.tool_choice, ChatCompletionNamedToolChoiceParam
    ):
        return None

    phase1_max_tokens = max_tokens - reserve
    if thinking_token_budget is not None:
        phase1_max_tokens = min(phase1_max_tokens, thinking_token_budget)
    if phase1_max_tokens <= 0:
        return None

    transition_ids: list[int] = []
    transition_text = envs.VLLM_HARMONY_FINAL_TRANSITION_TEXT
    if transition_text:
        transition_ids = list(
            tokenizer.encode(transition_text, add_special_tokens=False)
        )

    return FinalReservePlan(
        max_tokens=max_tokens,
        phase1_max_tokens=phase1_max_tokens,
        reserve=reserve,
        transition_ids=transition_ids,
    )


class HarmonyChannelTracker:
    """Follows the Harmony channel of a streamed assistant output.

    Read-only: it never changes what the engine generates. If the output
    leaves the Harmony grammar the tracker marks itself ``unknown`` and the
    request passes through unchanged (the ``HarmonyParser`` fallback then
    surfaces the text as content).
    """

    def __init__(self) -> None:
        self._parser = get_streamable_parser_for_assistant()
        self.unknown = False

    def feed(self, token_ids: Sequence[int]) -> None:
        if self.unknown:
            return
        for token_id in token_ids:
            try:
                self._parser.process(token_id)
            except HarmonyError:
                self.unknown = True
                return

    @property
    def channel(self) -> str | None:
        if self.unknown:
            return None
        return self._parser.current_channel

    @property
    def in_message_body(self) -> bool:
        return not self.unknown and self._parser.state == StreamState.CONTENT

    def forced_continuation_ids(
        self, transition_ids: Sequence[int] = ()
    ) -> list[int] | None:
        """Tokens to append before continuing, or ``None`` to pass through.

        - inside the analysis message: the transition text (if any) and the
          switch to the final channel;
        - inside the final (or plain commentary) message: nothing, the
          continuation simply keeps writing the answer;
        - anywhere else (message header, tool call in flight, off-grammar
          output): ``None``.
        """
        if not self.in_message_body:
            return None
        if self._parser.current_recipient is not None:
            return None
        channel = self._parser.current_channel
        if channel == "analysis":
            return [*transition_ids, *HARMONY_FORCE_FINAL_TOKEN_IDS]
        if channel in ("final", "commentary"):
            return []
        return None


def strip_finish(res: RequestOutput) -> RequestOutput:
    """Copy of a finished output presented as an ordinary delta."""
    out = copy.copy(res)
    out.finished = False
    out.outputs = [
        dataclasses.replace(o, finish_reason=None, stop_reason=None)
        for o in res.outputs
    ]
    return out


def make_forced_delta(
    base: RequestOutput, forced_ids: Sequence[int], text: str
) -> RequestOutput:
    """Synthetic delta carrying the forced switch tokens."""
    template = base.outputs[0]
    out = copy.copy(base)
    out.finished = False
    out.outputs = [
        CompletionOutput(
            index=template.index,
            text=text,
            token_ids=list(forced_ids),
            cumulative_logprob=template.cumulative_logprob,
            logprobs=None,
            finish_reason=None,
            stop_reason=None,
            lora_request=template.lora_request,
        )
    ]
    return out


def make_terminal_length_output(base: RequestOutput) -> RequestOutput:
    """Empty finishing delta with ``finish_reason='length'``.

    Used when the continuation could not be scheduled after part of the
    request was already streamed.
    """
    template = base.outputs[0]
    out = copy.copy(base)
    out.finished = True
    out.outputs = [
        CompletionOutput(
            index=template.index,
            text="",
            token_ids=[],
            cumulative_logprob=template.cumulative_logprob,
            logprobs=None,
            finish_reason="length",
            stop_reason=None,
            lora_request=template.lora_request,
        )
    ]
    return out


def merge_metrics(
    first: RequestStateStats | None, second: RequestStateStats | None
) -> RequestStateStats | None:
    """Timing of the whole request: queue, schedule and first token from the
    first engine request, last token and generation count over both."""
    if first is None:
        return second
    if second is None:
        return first
    return dataclasses.replace(
        first,
        num_generation_tokens=first.num_generation_tokens
        + second.num_generation_tokens,
        last_token_ts=second.last_token_ts or first.last_token_ts,
        is_corrupted=first.is_corrupted or second.is_corrupted,
    )


def rebase_output(
    res2: RequestOutput, base: RequestOutput, request_id: str
) -> RequestOutput:
    """Present a continuation output as part of the original request."""
    out = copy.copy(res2)
    out.request_id = request_id
    out.prompt = base.prompt
    out.prompt_token_ids = base.prompt_token_ids
    out.prompt_logprobs = base.prompt_logprobs
    out.encoder_prompt = base.encoder_prompt
    out.encoder_prompt_token_ids = base.encoder_prompt_token_ids
    out.num_cached_tokens = base.num_cached_tokens
    out.num_cache_creation_tokens = base.num_cache_creation_tokens
    out.lora_request = base.lora_request
    out.metrics = merge_metrics(base.metrics, res2.metrics)
    return out


def merge_final_outputs(
    res1: RequestOutput,
    forced_ids: Sequence[int],
    forced_text: str,
    res2: RequestOutput,
    request_id: str,
) -> RequestOutput:
    """One finished output covering both phases (non-streaming path)."""
    o1 = res1.outputs[0]
    o2 = res2.outputs[0]
    cumulative_logprob = (
        None
        if o1.cumulative_logprob is None or o2.cumulative_logprob is None
        else o1.cumulative_logprob + o2.cumulative_logprob
    )
    merged = CompletionOutput(
        index=o1.index,
        text=o1.text + forced_text + o2.text,
        token_ids=[*o1.token_ids, *forced_ids, *o2.token_ids],
        cumulative_logprob=cumulative_logprob,
        logprobs=None,
        routed_experts=None,
        finish_reason=o2.finish_reason,
        stop_reason=o2.stop_reason,
        lora_request=o1.lora_request,
    )
    out = rebase_output(res2, res1, request_id)
    out.outputs = [merged]
    out.finished = True
    return out
