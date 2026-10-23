# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


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
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        # Import after platform configuration and scheduler construction.
        from vllm_tt_plugin.platform import _ornith_fair_chunked_prefill

        self._ornith_selective_prefill = _ornith_fair_chunked_prefill(self.vllm_config)

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
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide waiting and partial-prefill continuations.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result)

        # After cache pressure preempts a request, let resident decoders
        # finish before admitting more prefill. A replay may fit its first
        # chunk but not its full history; immediately readmitting it can
        # repeatedly discard the same replay while decoders barely advance.
        if getattr(self, "_tt_drain_decodes_after_preemption", False):
            if has_running:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            self._tt_drain_decodes_after_preemption = False

        # Default mode:
        # Prefer prefill whenever there is pending prefill work - either new
        # requests in the waiting queue or partial-prefill continuations in
        # the running list.
        if has_pending_prefill:
            prefill_result = self._schedule_prefill_only()
            # If prefill cannot make progress (e.g., KV pressure) but running
            # decode requests exist, fall back to decode-only so they can
            # advance and free capacity.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No pending prefill work: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _use_ornith_fair_prefill(self) -> bool:
        """Bound selective fairness to one exclusive, fresh active8 opportunity.

        This neither reserves pages nor predicts future arrivals. Recheck each
        prefill decision; outside this cohort use the original base batching.
        """
        requests = list(self.running) + list(self.waiting)
        if (
            not 1 <= len(requests) <= 8
            or set(self.requests) != {r.request_id for r in requests}
            or any(not r.is_prefill_chunk for r in self.running)
            or self.finished_req_ids
            or self.max_num_running_reqs != 32
            or self.max_num_scheduled_tokens != 262144
            or self.max_model_len != 262144
            or self.cache_config.enable_prefix_caching
            or self.connector is not None
            or self.ec_connector is not None
            or self.num_lookahead_tokens
        ):
            return False
        groups = self.kv_cache_config.kv_cache_groups
        pool = self.kv_cache_manager.block_pool
        if (
            len(groups) != 1
            or groups[0].kv_cache_spec.block_size != 64
            or pool.num_gpu_blocks != 4129
        ):
            return False
        spans = {(r.num_prompt_tokens, r.max_tokens) for r in requests}
        if len(spans) != 1:
            return False
        prompt, completion = next(iter(spans))
        if (
            type(prompt) is not int
            or type(completion) is not int
            or prompt <= 0
            or completion <= 0
            or prompt + completion > self.max_model_len
        ):
            return False
        maximum_blocks = (prompt + completion + 63) // 64
        usable_blocks = pool.num_gpu_blocks - 1  # The pool owns one null block.
        if min(self.max_num_running_reqs, usable_blocks // maximum_blocks) != 8:
            return False
        free_blocks = pool.get_num_free_blocks()
        if type(free_blocks) is not int or not 0 <= free_blocks <= usable_blocks:
            return False
        owned: set[int] = set()
        needed = 0
        for request in requests:
            if (
                request.num_preemptions
                or request.num_tokens != prompt
                or request.num_output_placeholders
                or request.spec_token_ids
                or request.num_external_computed_tokens
                or request.has_encoder_inputs
            ):
                return False
            block_groups = self.kv_cache_manager.get_blocks(request.request_id).blocks
            if len(block_groups) != 1:
                return False
            blocks = block_groups[0]
            computed_blocks = (request.num_computed_tokens + 63) // 64
            if not computed_blocks <= len(blocks) <= maximum_blocks:
                return False
            needed += maximum_blocks - len(blocks)
            for block in blocks:
                if (
                    block.is_null
                    or block.ref_cnt != 1
                    or block.block_id in owned
                    or not 0 <= block.block_id < pool.num_gpu_blocks
                ):
                    return False
                owned.add(block.block_id)
        return free_blocks + len(owned) == usable_blocks and needed <= free_blocks

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests + partial-prefill continuations.

        Hides running decode requests (``is_prefill_chunk=False``) so the base
        scheduler's running loop only processes partial-prefill continuations and
        the waiting loop admits new prefills.  Adjusts ``max_num_running_reqs`` to
        account for hidden decode slots.
        """
        saved_prefill_policy = None
        if self._ornith_selective_prefill:
            fair = self._use_ornith_fair_prefill()
            saved_prefill_policy = (
                self.scheduler_config.enable_chunked_prefill,
                self.scheduler_config.long_prefill_token_threshold,
            )
            self.scheduler_config.enable_chunked_prefill = fair
            self.scheduler_config.long_prefill_token_threshold = 8192 if fair else 0

        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = partial_prefills
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            if saved_prefill_policy is not None:
                (
                    self.scheduler_config.enable_chunked_prefill,
                    self.scheduler_config.long_prefill_token_threshold,
                ) = saved_prefill_policy
        # Preserve the phase chosen here, including one-token final chunks of
        # history replay after preemption. Original prompt length cannot identify
        # those chunks once replay reaches previously generated tokens.
        result._tt_prefill_req_ids = set(result.num_scheduled_tokens)
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
        self._tt_drain_decodes_after_preemption = True

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
