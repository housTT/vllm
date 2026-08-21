# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request
from vllm_tt_plugin.config import get_tt_config
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

# ---------------------------------------------------------------- prefill admission cap
#
# How many requests one prefill step may carry. TT prefill is a *sequential* loop
# over the batch on the model side (one ``prefill_request_into_slot`` call per
# request), so a step that admits N requests costs N x T either way -- but vLLM
# publishes a step's sampled tokens only when the step returns, so all N first
# tokens are withheld until the last request's prefill finishes. Admitting one
# request per step releases each first token as soon as its own prefill completes:
# for a wave of N arriving together, mean TTFT drops from N x T to (N + 1) / 2 x T.
# Nothing about the computation changes -- same per-request prefill call, same
# state slot (see ``_schedule_prefill_only``).
#
# It is not free, and the cost grows with prompt length. The model's
# ``_ensure_traces_replay_safe`` runs at every prefill-step entry and re-captures
# the decode traces when the previous step compiled new programs, so splitting one
# wave into N steps moves up to N - 1 re-captures *inside* the wave. A re-capture's
# cost scales with the prefill program set, which scales with the number of 2048-token
# chunk offsets a prompt spans. That inserted time lands between each early request's
# first and second token, so it shows up as TPOT.
#
# Measured on 4x Blackhole p300c, batch 8, OSL 512, full occupancy:
#
#   ISL   1024:  TTFT median -42.9%,  TPOT +5.5%   -> worth it
#   ISL  65536:  TTFT median -17.7%,  TPOT +21.9%  -> not worth it
#
# Hence the length gate: the cap applies automatically only to prompts at or below
# ``prefill_cap_max_prompt_len``. See ``_PrefillCapPolicy`` for how the default
# threshold was derived and what it is still missing.
#
# ``cap = None`` means "no cap", i.e. filling the step with every admissible
# waiting request, which is the behaviour before this cap existed.
_PREFILL_CAP_ENV = "TT_MAX_PREFILLS_PER_STEP"
_PREFILL_CAP_KEY = "max_prefills_per_step"
DEFAULT_MAX_PREFILLS_PER_STEP = 1

_PREFILL_CAP_LEN_ENV = "TT_PREFILL_CAP_MAX_PROMPT_LEN"
_PREFILL_CAP_LEN_KEY = "prefill_cap_max_prompt_len"

# Sentinel threshold meaning "no prompt is above it", i.e. the cap always applies.
_UNBOUNDED_PROMPT_LEN = 1 << 62

# Threshold in prompt tokens, inclusive: at or below it the cap is on, above it off.
#
# None means no threshold: the cap applies at every prompt length. That is the
# default because the cap was measured to be free at every length tested.
#
# An earlier revision defaulted this to 4096, on a model that read the cap's cost
# off mean TPOT (+5.5% at ISL 1024, +21.9% at 65536) and set the threshold to keep
# that under 10%. The model was not wrong about mean TPOT -- a later measurement at
# ISL 8192 came in at +41.3%, steeper still. It was wrong that mean TPOT is a cost.
#
# vllm bench serve computes TPOT as (e2e_latency - TTFT) / (output_tokens - 1). A
# request that has finished its own prefill but sits behind other requests' prefills
# produces no tokens, and that idle time is charged to its TPOT. Capping admission
# starts each request's first token earlier, so each request spends more of its life
# in that waiting state -- mean TPOT rises while nothing gets slower. The time is
# moved from "waiting for the first token" into "gaps between tokens", and the total
# is conserved. Measured, cap ON vs OFF, batch 8, OSL 512, same session back-to-back:
#
#              ISL 8192 (n=48)        ISL 65536 (n=16)
#   e2e mean         +0.0%                  -0.1%
#   e2e p99          -0.1%                  -0.1%
#   aggregate        -0.0%                  +0.1%
#   median ITL       +0.2%                  +0.1%
#   TTFT median     -43.2%                 -10.8%
#   mean TPOT       +41.3%                 +37.4%
#
# So the cap buys a large TTFT improvement for no measurable change in end-to-end
# latency, throughput, or decode step time, and a 4096 threshold would switch that
# win off for every prompt above 4096 tokens.
#
# Set this to a token count to restore the threshold behaviour. The reason to do so
# is a deployment that values smooth streaming over fast first tokens: mean TPOT is
# not a decode metric, but choppier inter-token spacing is real and a user can feel
# it even when the request finishes at the same moment. Judge that with median ITL
# (unchanged here) alongside mean TPOT, never mean TPOT alone.
DEFAULT_PREFILL_CAP_MAX_PROMPT_LEN = _UNBOUNDED_PROMPT_LEN

# Model types whose TT implementation can restore a partial prefill's recurrent state after another
# request or a decode step used the device. Restricting this separately from platform.py's chunked
# prefill allow-list is intentional: splitting a prompt and *interleaving* its chunks are distinct
# statefulness contracts. The environment override exists for controlled A/Bs and emergency rollback.
_INTERLEAVED_PREFILL_MODEL_TYPES = {"qwen3_5_moe"}
_INTERLEAVE_ENV = "TT_INTERLEAVE_PREFILL_CHUNKS"


def _resolve_chunk_interleave(vllm_config) -> bool:
    scheduler_config = vllm_config.scheduler_config
    model_type = getattr(vllm_config.model_config.hf_config, "model_type", None)
    default = bool(
        scheduler_config.enable_chunked_prefill
        and model_type in _INTERLEAVED_PREFILL_MODEL_TYPES
    )
    raw = os.getenv(_INTERLEAVE_ENV)
    if raw is None:
        enabled = default
    else:
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            enabled = True
        elif text in ("0", "false", "no", "off"):
            enabled = False
        else:
            raise ValueError(
                f"{_INTERLEAVE_ENV} must be a boolean (0/1, false/true, off/on), got {raw!r}"
            )
    if enabled and not scheduler_config.enable_chunked_prefill:
        raise ValueError(
            f"{_INTERLEAVE_ENV}=1 requires enable_chunked_prefill=True; there are no chunks to interleave"
        )
    partial_limit = int(getattr(scheduler_config, "max_num_partial_prefills", 1))
    if enabled and partial_limit != 1:
        raise ValueError(
            "interleaved TT prefill uses one shared batch-1 recurrent-state pack and therefore "
            f"requires max_num_partial_prefills=1, got {partial_limit}"
        )
    return enabled


def _validate_chunk_interleave_admission(policy: "_PrefillCapPolicy") -> None:
    """Require one admitted prefill while a single batch-1 pack is the paused-state authority."""
    always_one = policy.cap == 1 and (
        policy.explicit or policy.max_prompt_len >= _UNBOUNDED_PROMPT_LEN
    )
    if not always_one:
        raise ValueError(
            "interleaved TT prefill requires max_prefills_per_step=1 with no prompt-length gate; "
            "set TT_INTERLEAVE_PREFILL_CHUNKS=0 before using a different admission policy"
        )


@dataclass(frozen=True)
class _PrefillCapPolicy:
    """Resolved prefill-admission policy.

    ``cap`` is the per-step admission ceiling, or ``None`` for no cap at all.
    ``explicit`` records that an operator set ``cap`` by hand, in which case it
    applies to every prompt and the length gate is bypassed entirely -- an explicit
    setting must beat anything chosen automatically. ``max_prompt_len`` is the
    automatic gate's threshold in prompt tokens, inclusive.
    """

    cap: int | None
    explicit: bool
    max_prompt_len: int

    def describe(self) -> str:
        if self.cap is None:
            return (
                "uncapped"
                if self.explicit
                else "uncapped (no cap configured)"
            )
        if self.explicit:
            return f"{self.cap} prefill(s) per step for every prompt (set explicitly)"
        if self.max_prompt_len >= _UNBOUNDED_PROMPT_LEN:
            return f"{self.cap} prefill(s) per step for every prompt (no threshold)"
        return (
            f"{self.cap} prefill(s) per step for prompts <= {self.max_prompt_len} "
            "tokens, uncapped above"
        )


def _parse_prefill_cap(raw: object, source: str) -> int | None:
    """Read a prefill-admission cap. ``0``/``""``/``none``/``off`` mean no cap."""
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("", "none", "off", "unlimited"):
            return None
        try:
            value = int(text)
        except ValueError:
            raise ValueError(
                f"{source} must be a non-negative integer "
                f'(or "none"/"off" for no cap), got {raw!r}'
            ) from None
    elif isinstance(raw, bool):
        # ``true`` is the natural way to ask for the default cap in JSON config.
        return DEFAULT_MAX_PREFILLS_PER_STEP if raw else None
    elif isinstance(raw, int):
        value = raw
    else:
        raise ValueError(f"{source} must be an integer, got {raw!r}")
    if value < 0:
        raise ValueError(f"{source} must be >= 0, got {raw!r}")
    return None if value == 0 else value


def _parse_prompt_len(raw: object, source: str) -> int:
    """Read the length gate's threshold. ``0`` disables the gate (cap everything)."""
    if isinstance(raw, bool):
        raise ValueError(f"{source} must be an integer token count, got {raw!r}")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("", "none", "off"):
            # "no threshold" means nothing is above it.
            return _UNBOUNDED_PROMPT_LEN
        try:
            value = int(text)
        except ValueError:
            raise ValueError(
                f"{source} must be a non-negative integer token count, got {raw!r}"
            ) from None
    elif isinstance(raw, int):
        value = raw
    else:
        raise ValueError(f"{source} must be an integer token count, got {raw!r}")
    if value < 0:
        raise ValueError(f"{source} must be >= 0, got {raw!r}")
    return value



def _read_setting(vllm_config, env: str, key: str) -> tuple[object | None, str]:
    """Read one setting from the environment, else the TT config namespace."""
    raw = os.getenv(env)
    if raw is not None:
        return raw, env
    try:
        raw = get_tt_config(vllm_config).get(key)
    except Exception:  # noqa: BLE001 - a malformed namespace is not ours to raise on
        raw = None
    return raw, f'additional_config["tt"]["{key}"]'


def resolve_prefill_cap_policy(vllm_config) -> _PrefillCapPolicy:
    """Resolve the prefill-admission policy.

    Precedence for each setting: the environment variable, then
    ``additional_config={"tt": {...}}``, then the default.

    - ``max_prefills_per_step`` / ``TT_MAX_PREFILLS_PER_STEP``: the per-step
      ceiling. Setting it explicitly (to any value, ``0`` included) makes it apply
      to every prompt regardless of length -- the operator override.
    - ``prefill_cap_max_prompt_len`` / ``TT_PREFILL_CAP_MAX_PROMPT_LEN``: the
      automatic gate's threshold in prompt tokens. Only consulted when the cap was
      *not* set explicitly. ``0`` disables the cap for every prompt; ``"none"``
      applies it to every prompt.
    """
    raw_cap, cap_source = _read_setting(vllm_config, _PREFILL_CAP_ENV, _PREFILL_CAP_KEY)
    raw_len, len_source = _read_setting(
        vllm_config, _PREFILL_CAP_LEN_ENV, _PREFILL_CAP_LEN_KEY
    )
    max_prompt_len = (
        DEFAULT_PREFILL_CAP_MAX_PROMPT_LEN
        if raw_len is None
        else _parse_prompt_len(raw_len, len_source)
    )
    if raw_cap is not None:
        return _PrefillCapPolicy(
            cap=_parse_prefill_cap(raw_cap, cap_source),
            explicit=True,
            max_prompt_len=max_prompt_len,
        )
    return _PrefillCapPolicy(
        cap=DEFAULT_MAX_PREFILLS_PER_STEP,
        explicit=False,
        max_prompt_len=max_prompt_len,
    )


@dataclass
class _PendingOutputs:
    """Counts a request's decode tokens that are still in the pipeline.

    Async scheduling starts new decode steps before the tokens from earlier
    steps have come back. If a request is preempted, it has to redo its prefill
    from scratch, and we have to throw the tokens still in the pipeline away instead
    of adding them to the output.

    ``outstanding`` counts tokens that were scheduled but have not come back
    yet. ``stale`` counts how many of those we have decided to throw away.
    """

    PENDING_ATTR: ClassVar[str] = "_tt_pending_outputs"

    outstanding: int = 0
    stale: int = 0

    def record(self) -> None:
        """Count one decode step; its token comes back a few steps later.

        A step counts once even if it speculates several tokens, because it
        still sends back a single output.
        """
        self.outstanding += 1

    def discard_outstanding(self) -> None:
        """On preempt, mark every token now in the pipeline to be thrown away."""
        self.stale = self.outstanding

    def is_next_stale(self) -> bool:
        """Take the next returned token; True means throw it away.

        Tokens come back in the order they were scheduled, so the stale ones
        always arrive before any fresh token from after the preempt.
        """
        if not self.outstanding:
            return False
        self.outstanding -= 1
        if not self.stale:
            return False
        self.stale -= 1
        return True

    @classmethod
    def for_request(cls, request: Request) -> "_PendingOutputs":
        pending = getattr(request, cls.PENDING_ATTR, None)
        if pending is None:
            pending = cls()
            setattr(request, cls.PENDING_ATTR, pending)
        return pending


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill: a long prefill may be split across scheduler
      steps.  After a partial chunk the request moves to ``running`` with
      ``is_prefill_chunk=True``; subsequent steps schedule the next chunk
      until the prompt is fully computed.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async_scheduling, preemption invalidates that request's
      scheduled-but-unreturned outputs (see ``_preempt_request``)

    Supports ``set_forced_mode`` for DP-gather coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when waiting is empty).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists (waiting queue or partial-prefill
      continuations), falling back to decode-only when prefill cannot make
      progress and running decode requests exist.

    Prefill admission is capped at ``max_prefills_per_step`` requests per step
    (default 1) so each request's first token is published by the step that
    prefilled it instead of being withheld until the rest of the wave finishes.
    The cap is applied automatically only to prompts at or below
    ``prefill_cap_max_prompt_len`` tokens, because its cost grows with prompt
    length while its benefit does not; an explicit operator setting bypasses that
    gate. See ``resolve_prefill_cap_policy``.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        self._prefill_cap = resolve_prefill_cap_policy(self.vllm_config)
        self._interleave_prefill_chunks = _resolve_chunk_interleave(self.vllm_config)
        if self._interleave_prefill_chunks:
            _validate_chunk_interleave_admission(self._prefill_cap)
        # Set after a scheduler step executes either a non-final chunk or the final continuation of
        # an already chunked prompt. If decode work is live, the next default-mode step pays that debt
        # before another prefill chunk, bounding a decoder's stall to one chunk at a time.
        self._decode_due_after_prefill_chunk = False
        # Last automatic decision, so a step with nothing new to admit (a
        # partial-prefill continuation on its own) keeps the wave's decision
        # rather than silently flipping, and so a flip can be logged when it
        # actually happens.
        self._auto_cap_active: bool | None = None
        logger.info(f"TT prefill admission: {self._prefill_cap.describe()}")
        logger.info(
            "TT prefill/decode chunk interleave: %s%s",
            "enabled" if self._interleave_prefill_chunks else "disabled",
            f" ({_INTERLEAVE_ENV})" if os.getenv(_INTERLEAVE_ENV) is not None else "",
        )

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request needs prefill work.

        True when the waiting queue is non-empty or any running request
        is a partial-prefill continuation (``is_prefill_chunk`` set by the
        base scheduler after the previous step).
        """
        return bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)

    def schedule(self) -> SchedulerOutput:
        has_pending_prefill = self._has_pending_prefill()
        has_running = any(not r.is_prefill_chunk for r in self.running)
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            result = self._schedule_prefill_with_interleave_tracking()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide waiting and partial-prefill continuations.
                result = self._schedule_decode_only()
                if result.total_num_scheduled_tokens:
                    self._decode_due_after_prefill_chunk = False
                return self._finalize_scheduler_output(result)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            if result.total_num_scheduled_tokens:
                self._decode_due_after_prefill_chunk = False
            return self._finalize_scheduler_output(result)

        # Default mode:
        # Prefer prefill whenever there is pending prefill work - either new
        # requests in the waiting queue or partial-prefill continuations in
        # the running list.
        if has_pending_prefill:
            if (
                self._interleave_prefill_chunks
                and self._decode_due_after_prefill_chunk
                and has_running
            ):
                decode_result = self._schedule_decode_only()
                if decode_result.total_num_scheduled_tokens:
                    self._decode_due_after_prefill_chunk = False
                    return self._finalize_scheduler_output(decode_result)

            prefill_result = self._schedule_prefill_with_interleave_tracking()
            # If prefill cannot make progress (e.g., KV pressure) but running
            # decode requests exist, fall back to decode-only so they can
            # advance and free capacity.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only()
                if result.total_num_scheduled_tokens:
                    self._decode_due_after_prefill_chunk = False
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No pending prefill work: run decode-only naturally.
        result = super().schedule()
        if result.total_num_scheduled_tokens:
            self._decode_due_after_prefill_chunk = False
        return self._finalize_scheduler_output(result)

    def _schedule_prefill_with_interleave_tracking(self) -> SchedulerOutput:
        """Schedule one pure-prefill step and remember whether a decode step is now due.

        Looking both before and after scheduling handles the two boundary cases. A newly admitted
        long prompt is marked partial only *after* its first chunk is scheduled; the final chunk of a
        continuation is partial only *before* it is scheduled. Either one must be followed by decode
        when other requests are already generating, otherwise two adjacent chunks can recreate the
        multi-second streaming stall this policy exists to remove.
        """
        had_partial = any(request.is_prefill_chunk for request in self.running)
        result = self._schedule_prefill_only()
        has_partial = any(request.is_prefill_chunk for request in self.running)
        if (
            self._interleave_prefill_chunks
            and result.total_num_scheduled_tokens
            and (had_partial or has_partial)
        ):
            self._decode_due_after_prefill_chunk = True
        return result

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _pending_prompt_len(self) -> int | None:
        """Prompt length of the request the waiting loop would admit next.

        One peek at the front of the waiting queue -- no scan, no allocation, and
        the same request the base scheduler is about to pop. ``None`` when nothing
        is waiting, which means this step has no new admission to gate.
        """
        if not self.waiting:
            return None
        try:
            return int(self.waiting.peek_request().num_prompt_tokens)
        except (IndexError, AttributeError):
            # An empty queue that still tested truthy, or a request type without
            # the attribute: neither is worth failing a scheduling step over.
            return None

    def _effective_prefill_cap(self) -> int | None:
        """Per-step prefill admission ceiling, ``None`` for no cap.

        An explicit operator setting wins unconditionally. Otherwise the cap is
        gated on the next waiting request's prompt length: at or below the
        threshold the cap pays for itself (measured -42.9% TTFT median for +5.5%
        TPOT at ISL 1024), above it the per-step trace re-capture cost dominates
        (-17.7% TTFT median for +21.9% TPOT at ISL 65536).
        """
        policy = self._prefill_cap
        if policy.cap is None or policy.explicit:
            return policy.cap

        length = self._pending_prompt_len()
        if length is None:
            # Nothing new to admit this step; hold the wave's current decision.
            active = self._auto_cap_active
            return policy.cap if active else None

        active = length <= policy.max_prompt_len
        if policy.max_prompt_len >= _UNBOUNDED_PROMPT_LEN:
            # No threshold: the decision can never flip, so say it once and plainly
            # rather than quoting a sentinel at whoever is reading the log.
            if self._auto_cap_active is None:
                logger.info(
                    f"TT prefill admission: capping admission at {policy.cap} "
                    f"prefill(s) per step for every prompt (no length threshold); "
                    f"first request is {length} token(s)"
                )
            self._auto_cap_active = True
            return policy.cap
        if active != self._auto_cap_active:
            side = "at or below" if active else "above"
            outcome = (
                f"capping admission at {policy.cap} prefill(s) per step"
                if active
                else "leaving admission uncapped"
            )
            which = "first request" if self._auto_cap_active is None else "pending prompt"
            switch = "" if self._auto_cap_active is None else "switching to "
            logger.info(
                f"TT prefill admission: {which} is {length} token(s), {side} the "
                f"{policy.max_prompt_len}-token threshold - {switch}{outcome}"
            )
            self._auto_cap_active = active
        return policy.cap if active else None

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests + partial-prefill continuations.

        Hides running decode requests (``is_prefill_chunk=False``) so the base
        scheduler's running loop only processes partial-prefill continuations and
        the waiting loop admits new prefills.  Adjusts ``max_num_running_reqs`` to
        account for hidden decode slots.

        ``max_num_running_reqs`` is the only gate on the base scheduler's waiting
        loop (``Scheduler.schedule`` breaks out of it once ``len(self.running)``
        reaches it), so lowering it here is also how the per-step prefill cap is
        applied: with ``running`` reduced to the partial-prefill continuations, a
        ceiling of ``cap`` admits at most ``cap`` new prefills. Two properties the
        base scheduler's ``assert len(self.running) <= self.max_num_running_reqs``
        depends on:

        - the ceiling is never lowered below ``len(partial_prefills)``, which is
          what ``self.running`` already holds on entry (the running loop must be
          free to advance every continuation it was given);
        - it is never raised above the free-slot count, so the cap can only ever
          admit *fewer* requests than the uncapped policy, never more.

        ``_effective_prefill_cap`` decides whether the cap applies at all for the
        prompt about to be admitted; ``None`` from it restores the pre-cap policy
        line for line.

        Admitting one at a time does not change which slot a request gets. The
        runner's ``_alloc_prefill_state_slots`` gives local row ``j`` the ``j``-th
        smallest slot not held off-batch, and it commits each assignment to
        ``_req_state_slot`` before the next step reads it as held -- so N requests
        admitted over N steps land on exactly the slots the one batched step would
        have given them, in the same FCFS order. See the slot-remap note in the
        stage write-up.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        prefill_capacity = max(0, saved_max - len(pure_decodes))
        if self._interleave_prefill_chunks and partial_prefills:
            # A partial Ornith request owns the model's one batch-1 prefill pack while decode runs.
            # Do not admit a second prefill that would replace that authority before the continuation
            # has consumed it. This remains one even if an operator raises max_prefills_per_step.
            prefill_capacity = len(partial_prefills)
        else:
            cap = self._effective_prefill_cap()
            if cap is not None:
                prefill_capacity = max(len(partial_prefills), min(prefill_capacity, cap))
        self.running = partial_prefills
        self.max_num_running_reqs = prefill_capacity
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Hides the waiting queue **and** any partial-prefill continuations
        so the base scheduler only sees decode-phase requests.  Preempted
        requests are merged back into the original waiting queue afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]

        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)

        return result

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and drop any decode tokens still in the pipeline.

        The base class frees the request's KV cache and queues it to redo its
        prefill. Under async scheduling, the tokens it already scheduled are
        still on their way back; those were built on the now-freed cache, so we
        mark them to be thrown away and reset the placeholder count so the base
        scheduler treats the request as a fresh prefill.
        """
        super()._preempt_request(request, timestamp)

        if not self.scheduler_config.async_scheduling:
            return

        _PendingOutputs.for_request(request).discard_outstanding()
        request.num_output_placeholders = 0

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """After scheduling, count the decode tokens this step put in the pipeline.

        Only decode steps produce a token to track; prefill chunks do not. This
        running count is what lets a later preempt know how many in-flight
        tokens it has to throw away.
        """
        super()._update_after_schedule(scheduler_output)

        if self.scheduler_config.async_scheduling:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests[req_id]
                if not request.is_prefill_chunk:
                    _PendingOutputs.for_request(request).record()

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Add a returned token to the request, unless a preempt invalidated it.

        When a token comes back for a request that was preempted while the token
        was in the pipeline, adding it would corrupt the output, so we throw it
        away here instead of handing it to the base class.
        """
        if not self.scheduler_config.async_scheduling:
            return super()._update_request_with_output(request, new_token_ids)

        if _PendingOutputs.for_request(request).is_next_stale():
            request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids)
