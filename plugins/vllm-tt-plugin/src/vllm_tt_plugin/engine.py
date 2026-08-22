# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import pickle
import queue
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig, VllmConfig
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request
from vllm_tt_plugin.config import get_tt_config, get_tt_per_lane_max_num_seqs
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.scheduler import TTSchedulingMode

logger = init_tt_logger(__name__)
_T = TypeVar("_T")

_DEFAULT_INPUT_QUEUE_BATCHING_DELAY = 0.002
_ORNITH_GROUPED_INPUT_QUEUE_BATCHING_DELAY = 0.250
_ORNITH_THROUGHPUT_INPUT_QUEUE_BATCHING_DELAY = 2.000
_ORNITH_PREFILL_PROFILE_ENV_VAR = "ORNITH_VLLM_PREFILL_PROFILE"
_ORNITH_PREFILL_PROFILES = ("interactive", "throughput")


def _get_ornith_prefill_profile() -> str:
    """Resolve the live latency/throughput trade-off for grouped Ornith prefills."""

    raw = os.environ.get(_ORNITH_PREFILL_PROFILE_ENV_VAR, "").strip()
    if not raw:
        return "interactive"
    if raw not in _ORNITH_PREFILL_PROFILES:
        choices = ", ".join(_ORNITH_PREFILL_PROFILES)
        raise ValueError(
            f"{_ORNITH_PREFILL_PROFILE_ENV_VAR} must be one of {choices}; got {raw!r}"
        )
    return raw


def _get_grouped_ornith_prefill_target(vllm_config: VllmConfig) -> int | None:
    """Return the synchronized Ornith prefill target, when it is available."""
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    scheduler_config = vllm_config.scheduler_config
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 1))
    max_num_partial_prefills = int(
        getattr(scheduler_config, "max_num_partial_prefills", 1)
    )
    if (
        model_type == "qwen3_5_moe"
        and max_num_seqs > 1
        and max_num_partial_prefills > 1
    ):
        return min(max_num_seqs, max_num_partial_prefills)
    return None


def _get_input_queue_batching_delay(vllm_config: VllmConfig) -> float:
    """Resolve the idle input-queue coalescing delay for this engine.

    Ornith can execute up to four long-prefill rows in one synchronized device
    batch, but tokenization can deliver otherwise-concurrent requests to the
    engine several milliseconds apart. Give that model a wider default
    coalescing window only when both scheduler limits permit grouped prefills.
    The gathered-DP handler retains its historical 2 ms default for B1 and
    other model families. The single-engine handler uses this resolver only
    for grouped Ornith or when the user explicitly supplies the TT setting.
    A user-provided value is always authoritative, including zero to disable
    the delay.
    """
    tt_config = get_tt_config(vllm_config)
    if "input_queue_batching_delay" in tt_config:
        return float(tt_config["input_queue_batching_delay"])

    if _get_grouped_ornith_prefill_target(vllm_config) is not None:
        if _get_ornith_prefill_profile() == "throughput":
            return _ORNITH_THROUGHPUT_INPUT_QUEUE_BATCHING_DELAY
        return _ORNITH_GROUPED_INPUT_QUEUE_BATCHING_DELAY
    return _DEFAULT_INPUT_QUEUE_BATCHING_DELAY


def _normal_init_dp_group(parallel_config: ParallelConfig) -> dist.ProcessGroup:
    """Create the TT engine DP group with rooted collectives enabled."""
    from torch.distributed import DistNetworkError

    if dist.is_initialized():
        raise RuntimeError(
            "TT DP gather requires a fresh default torch.distributed process "
            "group in the engine process."
        )

    max_retries = 5
    last_exc: Exception | None = None
    for _ in range(max_retries):
        init_method = get_tcp_uri(
            parallel_config.data_parallel_master_ip,
            parallel_config.get_next_dp_init_port(),
        )
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=parallel_config.data_parallel_rank,
                world_size=parallel_config.data_parallel_size,
            )
            return dist.group.WORLD
        except DistNetworkError as e:
            if "EADDRINUSE" in str(e):
                logger.warning("Address already in use. Retrying with a new port.")
                last_exc = e
                continue
            raise

    assert last_exc is not None
    raise last_exc


@dataclass
class DPGatherHandle:
    future: Future[tuple[torch.Tensor, list]]
    scheduler_output: SchedulerOutput | None
    local_has_requests: bool
    is_decode: bool
    overlap_ok: bool
    any_needs_logprobs: bool
    intermediate_prefill_mask: torch.Tensor | None
    req_ids: list[str]
    req_id_to_index: dict[str, int]


def _process_tt_input_queue(
    engine_core: EngineCoreProc,
    *,
    poll_idle_queue: bool,
) -> None:
    """Process TT client input and coalesce an idle prefill wave.

    ``EngineCoreProc`` normally blocks until the first request and then drains
    only messages that have already reached its input queue.  That is too
    early for grouped Ornith prefills: concurrent HTTP requests can finish
    tokenization a few milliseconds apart, after the first hardware step has
    already started.  This helper retains each engine's idle behavior, then
    uses an event-driven, absolute coalescing deadline once a request is
    scheduler-visible.

    Gathered-DP ranks must periodically leave the idle queue to progress
    collectives, whereas a single engine can preserve the base class's
    indefinite idle wait.  ``poll_idle_queue`` selects only that distinction;
    FIFO draining and coalescing are shared by both paths.
    """
    delay = _get_input_queue_batching_delay(engine_core.vllm_config)
    grouped_target = _get_grouped_ornith_prefill_target(engine_core.vllm_config)
    profile = (
        _get_ornith_prefill_profile()
        if grouped_target is not None
        else "not_applicable"
    )
    waited = False

    if poll_idle_queue:
        idle_timed_out = False
        # Before a request is scheduler-visible, retain the historical 2 ms
        # DP polling cadence. The wider grouped deadline starts only after a
        # request exists and therefore cannot park otherwise-idle peer ranks.
        idle_timeout = min(max(delay, 0.0), _DEFAULT_INPUT_QUEUE_BATCHING_DELAY)
        idle_deadline = time.monotonic() + idle_timeout
        while (
            not engine_core.engines_running
            and not engine_core.scheduler.has_requests()
            and not engine_core.batch_queue
            and not getattr(engine_core, "_dp_in_flight", None)
            and not engine_core._scheduler_paused
        ):
            # A bounded blocking read wakes immediately on arrival while idle
            # TT ranks still return to the gathered-DP collective loop.
            try:
                if idle_timeout > 0:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        idle_timed_out = True
                        break
                    req = engine_core.input_queue.get(timeout=remaining)
                else:
                    req = engine_core.input_queue.get_nowait()
                engine_core._handle_client_request(*req)
                waited = True
            except queue.Empty:
                idle_timed_out = True
                break

        if idle_timed_out:
            if waited:
                logger.debug("EngineCore loop active.")
            return
    else:
        # Preserve EngineCoreProc's idle semantics for DP=1, including abort
        # queue cleanup and an indefinite blocking read before the first
        # scheduler-visible request.
        while (
            not engine_core.engines_running
            and not engine_core.scheduler.has_requests()
            and not engine_core.batch_queue
            and not engine_core._scheduler_paused
        ):
            if engine_core.input_queue.empty():
                with engine_core.aborts_queue.mutex:
                    engine_core.aborts_queue.queue.clear()
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = engine_core.input_queue.get()
            engine_core._handle_client_request(*req)

    if waited:
        logger.debug("EngineCore loop active.")

    # Use one absolute deadline for the whole coalescing phase. Arrivals wake
    # the queue read immediately, but request/control-message trickle cannot
    # extend the batching window indefinitely.
    coalescing_started = time.monotonic()
    coalescing_deadline = coalescing_started + max(delay, 0.0)
    max_num_seqs = int(engine_core.vllm_config.scheduler_config.max_num_seqs)
    exit_reason = "queue_drained"
    while True:
        num_running, num_waiting = engine_core.scheduler.get_request_counts()
        coalescing_target = (
            grouped_target if grouped_target is not None else max_num_seqs
        )
        target_reached = num_waiting >= coalescing_target
        has_pending_engine_work = bool(
            engine_core.engines_running
            or engine_core.batch_queue
            or getattr(engine_core, "_dp_in_flight", None)
            or engine_core._scheduler_paused
        )
        should_wait = (
            delay > 0
            and num_waiting > 0
            and num_running == 0
            and not target_reached
            and not has_pending_engine_work
        )
        try:
            if should_wait:
                remaining = coalescing_deadline - time.monotonic()
                if remaining > 0:
                    try:
                        req = engine_core.input_queue.get(timeout=remaining)
                    except queue.Empty:
                        # Close the timeout boundary race: an abort, executor
                        # failure, or request may have arrived as the blocking
                        # read expired. Drain it below before a hardware step.
                        req = engine_core.input_queue.get_nowait()
                else:
                    # The deadline stops further blocking, not FIFO handling
                    # of messages that are already queued.
                    req = engine_core.input_queue.get_nowait()
            else:
                # Never wait once the target is full or other engine work is
                # active, but drain all messages already queued in FIFO order.
                req = engine_core.input_queue.get_nowait()
        except queue.Empty:
            if should_wait and time.monotonic() >= coalescing_deadline:
                exit_reason = "deadline"
            elif target_reached:
                exit_reason = "target_reached"
            elif has_pending_engine_work:
                exit_reason = "engine_work_pending"
            elif num_waiting == 0:
                exit_reason = "no_waiting_requests"
            break
        engine_core._handle_client_request(*req)

    num_running, num_waiting = engine_core.scheduler.get_request_counts()
    tt_config = get_tt_config(engine_core.vllm_config)
    if num_waiting > 0 and (
        grouped_target is not None or "input_queue_batching_delay" in tt_config
    ):
        logger.info(
            "TT prefill coalescing exit profile=%s delay_seconds=%.3f "
            "elapsed_seconds=%.3f target=%d waiting=%d running=%d reason=%s",
            profile,
            delay,
            time.monotonic() - coalescing_started,
            grouped_target if grouped_target is not None else max_num_seqs,
            num_waiting,
            num_running,
            exit_reason,
        )


class TTEngineCoreProc(EngineCoreProc):
    """TT single-engine core with opt-in or grouped-prefill coalescing."""

    def _process_input_queue(self) -> None:
        tt_config = get_tt_config(self.vllm_config)
        grouped_ornith = (
            _get_grouped_ornith_prefill_target(self.vllm_config) is not None
        )
        if grouped_ornith or "input_queue_batching_delay" in tt_config:
            _process_tt_input_queue(self, poll_idle_queue=False)
        else:
            # Do not introduce a queue delay for unrelated DP=1 models. Their
            # established behavior is the upstream blocking/draining loop;
            # users can explicitly opt in through the TT config key above.
            EngineCoreProc._process_input_queue(self)


class TTDPEngineCoreProc(DPEngineCoreProc):
    """TT data-parallel engine core with gathered-batch orchestration."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DEBUG_DPG = os.environ.get("DP_GATHER_DEBUG") == "1"

        def dlog_logger(msg: str, *a: object) -> None:
            if DEBUG_DPG:
                formatted = (msg % a) if a else msg
                logger.info("dp_gather r%d: %s", self.dp_rank, formatted)

        self.dlog = dlog_logger
        super().__init__(vllm_config, *args, **kwargs)
        self._dp_in_flight: DPGatherHandle | None = None
        if self.batch_queue is not None:
            self.step_fn = self.step_dp_with_batch_queue

    def _process_input_queue(self) -> None:
        _process_tt_input_queue(self, poll_idle_queue=True)

    def _init_tt_dp_group(self, parallel_config: ParallelConfig) -> None:
        self.dp_group = _normal_init_dp_group(parallel_config)

        local_dp_rank = parallel_config.data_parallel_rank_local
        dp_size = parallel_config.data_parallel_size
        local_dp_rank_tensor = torch.tensor(
            [local_dp_rank], dtype=torch.int32, device="cpu"
        )
        gathered_local_ranks = [
            torch.zeros(1, dtype=torch.int32) for _ in range(dp_size)
        ]
        dist.all_gather(gathered_local_ranks, local_dp_rank_tensor, group=self.dp_group)
        self.dp_device_ranks = [
            i
            for i, rank_tensor in enumerate(gathered_local_ranks)
            if rank_tensor.item() == 0
        ]
        logger.info("DP device ranks: %s", self.dp_device_ranks)

    def _init_data_parallel(self, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self._init_tt_dp_group(parallel_config)

    def shutdown(self) -> None:
        EngineCoreProc.shutdown(self)
        if dp_group := getattr(self, "dp_group", None):
            dist.destroy_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        start_wave = False
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                start_wave = True

        if self.has_coordinator and not self.engines_running:
            # The front-end normally notifies the coordinator before sending
            # the first request in a new wave. If that notification races with
            # wave completion state, this rank must still wake its peers before
            # entering TT gathered-DP collectives.
            self.engines_running = True
            start_wave = True

        if start_wave:
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

        super().add_request(request, request_wave)

    def run_busy_loop(self) -> None:
        while True:
            # Rendezvous all DP ranks at iteration start to prevent
            # FIFO-collective skew accumulation across iterations.
            # gloo collectives are matched in call order per group, so once
            # ranks drift by one iteration, every subsequent collective can
            # deadlock waiting for a future peer call.
            try:
                dist.barrier(group=self.dp_group)
            except RuntimeError as e:
                if "Connection closed by peer" in str(e):
                    raise SystemExit() from e
                raise

            self._process_input_queue()
            self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            # TT does not call execute_dummy_batch() on idle steps because
            # _dp_any_rank_has_scheduler_requests() already synchronises all
            # ranks before any execution is attempted.  Rank alignment for the
            # wave-finish all-reduce happens inside _has_global_unfinished_reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                    if self.has_coordinator and client_index == -1:
                        self.output_queue.put_nowait(
                            (
                                0,
                                EngineCoreOutputs(
                                    wave_complete=self.current_wave,
                                    scheduler_stats=self.scheduler.make_stats(),
                                ),
                            )
                        )
                self.current_wave += 1
                self.step_counter = 0

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        dist.destroy_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self._init_tt_dp_group(parallel_config)
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("TTDPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )

    def _dp_any_rank_has_scheduler_requests(self) -> bool:
        local_has_requests = 1 if self.scheduler.has_requests() else 0
        has_requests_t = torch.tensor([local_has_requests], dtype=torch.int32)
        try:
            dist.all_reduce(has_requests_t, op=dist.ReduceOp.SUM, group=self.dp_group)
        except RuntimeError as e:
            # During shutdown, peers may close connections mid-collective.
            if "Connection closed by peer" in str(e):
                logger.debug("Collective failed during shutdown, exiting gracefully")
                raise SystemExit() from e
            raise
        return int(has_requests_t.item()) > 0

    def _dp_negotiate_forced_mode(self) -> TTSchedulingMode:
        running = getattr(self.scheduler, "running", [])
        has_running = bool(running)
        has_waiting = bool(getattr(self.scheduler, "waiting", False))
        max_running = getattr(self.scheduler, "max_num_running_reqs", 0)
        has_partial_prefill = any(request.is_prefill_chunk for request in running)
        has_capacity = len(running) < max_running
        local_prefill_intent = (
            1
            if (
                has_partial_prefill
                or (has_waiting and ((not has_running) or has_capacity))
            )
            else 0
        )
        intent_tensor = torch.tensor([local_prefill_intent], dtype=torch.int32)
        self.dlog("before_intent_allreduce intent_tensor=%s", intent_tensor)
        dist.all_reduce(intent_tensor, op=dist.ReduceOp.MAX, group=self.dp_group)
        forced_mode = TTSchedulingMode.from_prefill_intent(int(intent_tensor.item()))
        self.dlog("after_intent_allreduce forced_mode=%s", forced_mode)
        self._dp_gather_forced_mode = forced_mode
        return forced_mode

    def _dp_apply_forced_mode(self, forced_mode: TTSchedulingMode) -> None:
        set_mode = getattr(self.scheduler, "set_forced_mode", None)
        if callable(set_mode):
            set_mode(forced_mode)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        if self._scheduler_paused:
            return {}, False

        if not self._dp_any_rank_has_scheduler_requests():
            return {}, False

        forced_mode = self._dp_negotiate_forced_mode()
        if not self.scheduler.has_requests():
            _ = self._execute_model_dp_gather(None, None)
            return {}, False

        self._dp_apply_forced_mode(forced_mode)
        scheduler_output = self.scheduler.schedule()
        self._dp_apply_forced_mode(TTSchedulingMode.DEFAULT)

        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        model_output = self._execute_model_dp_gather(scheduler_output, grammar_output)
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def step_dp_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        assert self.batch_queue is not None

        global_has_requests = self._dp_any_rank_has_scheduler_requests()
        prev_handle = self._dp_in_flight
        if not global_has_requests and prev_handle is None:
            return {}, False

        forced_mode = TTSchedulingMode.DEFAULT
        scheduler_output: SchedulerOutput | None = None
        grammar_output: GrammarOutput | None = None
        model_executed = False
        current_overlap_ok = False
        if global_has_requests:
            forced_mode = self._dp_negotiate_forced_mode()
            if self.scheduler.has_requests():
                self._dp_apply_forced_mode(forced_mode)
                scheduler_output = self.scheduler.schedule()
                self._dp_apply_forced_mode(TTSchedulingMode.DEFAULT)
                if not self.is_ec_producer:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0
                if not scheduler_output.pending_structured_output_tokens:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
            if forced_mode == TTSchedulingMode.DECODE_ONLY:
                current_overlap_ok = self._dp_can_attempt_steady_decode_from_scheduler(
                    scheduler_output, grammar_output
                )

        def _finalize_previous(
            handle: DPGatherHandle,
        ) -> dict[int, EngineCoreOutputs]:
            model_output = self.dp_gather_finalize(handle)
            if handle.scheduler_output is None:
                return {}
            return self.scheduler.update_from_output(
                handle.scheduler_output, model_output
            )

        # Always finalize the previous step before submitting the next one.
        #
        # The submit reads ``input_batch.token_ids_cpu`` to build the decode
        # input for the next step; that table is only updated once
        # ``apply_dp_execution_result`` runs inside ``_finalize_previous``. The
        # original overlap path (submit-next then finalize-prev) therefore
        # built the next step's input from stale token state, so the device
        # re-sampled the previous step's near-deterministic position — most
        # visibly as doubled ``<|end|>`` and ``<|start|>assistant`` tokens,
        # which break harmony parsing and silently null out chat responses.
        finalize_before_submit = prev_handle is not None

        engine_core_outputs: dict[int, EngineCoreOutputs] | None = {}
        if finalize_before_submit:
            assert prev_handle is not None
            engine_core_outputs = _finalize_previous(prev_handle)
            prev_handle = None

        if (
            scheduler_output is not None
            and grammar_output is None
            and scheduler_output.pending_structured_output_tokens
        ):
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

        next_handle: DPGatherHandle | None = None
        if global_has_requests:
            next_handle = self.dp_gather_submit(
                scheduler_output,
                grammar_output,
                overlap_ok=current_overlap_ok,
            )

        if not finalize_before_submit and prev_handle is not None:
            engine_core_outputs = _finalize_previous(prev_handle)

        self._dp_in_flight = next_handle

        if not global_has_requests:
            return engine_core_outputs, False

        return engine_core_outputs, model_executed

    def _dp_can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        local_overlap_ok = int(
            self.model_executor.collective_rpc(
                "can_attempt_steady_dp_decode_from_scheduler",
                args=(scheduler_output, grammar_output),
            )[0]
        )
        overlap_ok_t = torch.tensor([local_overlap_ok], dtype=torch.int32)
        dist.all_reduce(overlap_ok_t, op=dist.ReduceOp.MIN, group=self.dp_group)
        overlap_ok = bool(overlap_ok_t.item())
        self.dlog("steady_decode_overlap_ok=%s", overlap_ok)
        return overlap_ok

    def dp_gather_submit(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
        *,
        overlap_ok: bool = False,
    ) -> DPGatherHandle:
        parallel_config = self.vllm_config.parallel_config
        group = self.dp_group
        rank = self.dp_rank
        local_rank = parallel_config.data_parallel_rank_local
        world = parallel_config.data_parallel_size

        local_has_requests = scheduler_output is not None
        if scheduler_output is not None:
            self.dlog(
                "enter_gather tokens=%d",
                scheduler_output.total_num_scheduled_tokens,
            )

        assert hasattr(self, "_dp_gather_forced_mode"), "forced_mode not set"
        is_decode = self._dp_gather_forced_mode == TTSchedulingMode.DECODE_ONLY

        all_local_inputs = self.model_executor.collective_rpc(
            "build_dp_model_input", args=(scheduler_output, grammar_output)
        )[0]
        (
            local_input,
            local_max_blocks,
            local_has_structured,
            local_has_penalties,
            local_reset_batch,
            local_can_sample_device,
            local_needs_logprobs,
            intermediate_prefill_mask,
            req_ids,
            req_id_to_index,
        ) = all_local_inputs
        max_blocks_decode = None
        any_structured_inputs = False
        any_needs_logprobs = False

        gathered_inputs: Any = None
        if is_decode:
            input_info_t = torch.tensor(
                [
                    local_max_blocks,
                    local_has_structured,
                    local_has_penalties,
                    local_reset_batch,
                    1 - local_can_sample_device,
                    local_needs_logprobs,
                ],
                dtype=torch.int32,
            )
            dist.all_reduce(input_info_t, op=dist.ReduceOp.MAX, group=group)
            max_blocks_decode = int(input_info_t[0].item())
            any_structured_inputs = input_info_t[1].item() > 0
            any_penalties_inputs = input_info_t[2].item() > 0
            any_reset_batch = input_info_t[3].item() > 0
            all_sample_device = input_info_t[4].item() == 0
            any_needs_logprobs = input_info_t[5].item() > 0

            decode_inputs: dict[str, Any] = self.model_executor.collective_rpc(
                "build_dp_decode_gather_input",
                args=(
                    local_input,
                    max_blocks_decode,
                    any_structured_inputs,
                    any_penalties_inputs,
                ),
            )[0]

            int_local = decode_inputs["int_inputs"]
            float_local = decode_inputs["float_inputs"]

            stacked_int = None
            stacked_float = None
            gather_list_int = None
            gather_list_float = None
            if rank == 0:
                stacked_int = torch.empty(
                    (world, *int_local.shape), dtype=int_local.dtype
                )
                stacked_float = torch.empty(
                    (world, *float_local.shape), dtype=float_local.dtype
                )
                gather_list_int = [stacked_int[i] for i in range(world)]
                gather_list_float = [stacked_float[i] for i in range(world)]

            dist.gather(int_local, gather_list_int, dst=0, group=group)
            dist.gather(float_local, gather_list_float, dst=0, group=group)
            if len(self.dp_device_ranks) > 1:
                if rank == 0:
                    for dst in self.dp_device_ranks[1:]:
                        dist.send(stacked_int, dst=dst, group=group)
                        dist.send(stacked_float, dst=dst, group=group)
                elif local_rank == 0:
                    stacked_int = torch.empty(
                        (world, *int_local.shape), dtype=int_local.dtype
                    )
                    stacked_float = torch.empty(
                        (world, *float_local.shape), dtype=float_local.dtype
                    )
                    dist.recv(stacked_int, src=0, group=group)
                    dist.recv(stacked_float, src=0, group=group)

            gathered_tokens_inputs = None
            if any_penalties_inputs and (not all_sample_device or any_reset_batch):
                if rank == 0:
                    gathered_tokens_inputs = [None for _ in range(world)]
                local_tokens_inputs = decode_inputs["sampling_tokens_inputs"]
                dist.gather_object(
                    local_tokens_inputs, gathered_tokens_inputs, dst=0, group=group
                )

                if len(self.dp_device_ranks) > 1:
                    if rank == 0:
                        pickled_tokens = pickle.dumps(gathered_tokens_inputs)
                        tokens_tensor = torch.frombuffer(
                            pickled_tokens, dtype=torch.uint8
                        )
                        tokens_size = torch.tensor(
                            [tokens_tensor.numel()], dtype=torch.long
                        )
                        for dst in self.dp_device_ranks[1:]:
                            dist.send(tokens_size, dst=dst, group=group)
                            dist.send(tokens_tensor, dst=dst, group=group)
                    elif local_rank == 0:
                        tokens_size = torch.zeros(1, dtype=torch.long)
                        dist.recv(tokens_size, src=0, group=group)
                        tokens_tensor = torch.empty(
                            tokens_size.item(), dtype=torch.uint8
                        )
                        dist.recv(tokens_tensor, src=0, group=group)
                        gathered_tokens_inputs = pickle.loads(
                            tokens_tensor.numpy().tobytes()
                        )

            gathered_host_only_sample_params = None
            if not all_sample_device:
                if rank == 0:
                    gathered_host_only_sample_params = [None for _ in range(world)]
                local_host_only_sample_params = decode_inputs.get(
                    "host_only_sample_params"
                )
                dist.gather_object(
                    local_host_only_sample_params,
                    gathered_host_only_sample_params,
                    dst=0,
                    group=group,
                )

                if len(self.dp_device_ranks) > 1:
                    if rank == 0:
                        pickled_host_only = pickle.dumps(
                            gathered_host_only_sample_params
                        )
                        host_only_tensor = torch.frombuffer(
                            pickled_host_only, dtype=torch.uint8
                        )
                        host_only_size = torch.tensor(
                            [host_only_tensor.numel()], dtype=torch.long
                        )
                        for dst in self.dp_device_ranks[1:]:
                            dist.send(host_only_size, dst=dst, group=group)
                            dist.send(host_only_tensor, dst=dst, group=group)
                    elif local_rank == 0:
                        host_only_size = torch.zeros(1, dtype=torch.long)
                        dist.recv(host_only_size, src=0, group=group)
                        host_only_tensor = torch.empty(
                            host_only_size.item(), dtype=torch.uint8
                        )
                        dist.recv(host_only_tensor, src=0, group=group)
                        gathered_host_only_sample_params = pickle.loads(
                            host_only_tensor.numpy().tobytes()
                        )

            if local_rank == 0:
                gathered_inputs = {
                    "int_inputs": stacked_int,
                    "float_inputs": stacked_float,
                    "sampling_tokens_inputs": gathered_tokens_inputs,
                    "host_only_sample_params": gathered_host_only_sample_params,
                    "reset_batch": any_reset_batch,
                    "all_sample_device": all_sample_device,
                }

        else:
            gathered_inputs = None
            if rank == 0:
                gathered_inputs = [None for _ in range(world)]

            logprobs_flag_t = torch.tensor([local_needs_logprobs], dtype=torch.int32)
            dist.all_reduce(logprobs_flag_t, op=dist.ReduceOp.MAX, group=group)
            any_needs_logprobs = logprobs_flag_t[0].item() > 0

            dist.gather_object(local_input, gathered_inputs, dst=0, group=group)
            if len(self.dp_device_ranks) > 1:
                if rank == 0:
                    pickled_data = pickle.dumps(gathered_inputs)
                    object_tensor = torch.frombuffer(pickled_data, dtype=torch.uint8)
                    size_tensor = torch.tensor(
                        [object_tensor.numel()], dtype=torch.long
                    )
                    for dst in self.dp_device_ranks[1:]:
                        dist.send(size_tensor, dst=dst, group=group)
                        dist.send(object_tensor, dst=dst, group=group)
                elif local_rank == 0:
                    size_tensor = torch.zeros(1, dtype=torch.long)
                    dist.recv(size_tensor, src=0, group=group)
                    object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
                    dist.recv(object_tensor, src=0, group=group)
                    gathered_inputs = pickle.loads(object_tensor.numpy().tobytes())
        self.dlog("after_inputs_gather")

        should_submit = is_decode or (
            isinstance(gathered_inputs, list)
            and any(x is not None for x in gathered_inputs)
        )
        if should_submit:
            collective_future = cast(
                Future[list[tuple[torch.Tensor, list]]],
                self.model_executor.collective_rpc(
                    "concat_and_execute_dp",
                    args=(
                        gathered_inputs,
                        is_decode,
                        max_blocks_decode,
                        any_structured_inputs,
                    ),
                    kwargs={"non_block": True},
                    non_block=True,
                ),
            )
            future = _unwrap_single_worker_future(collective_future)
        else:
            future = self._completed_dp_gather_future()

        return DPGatherHandle(
            future=future,
            scheduler_output=scheduler_output,
            local_has_requests=local_has_requests,
            is_decode=is_decode,
            overlap_ok=overlap_ok,
            any_needs_logprobs=any_needs_logprobs,
            intermediate_prefill_mask=intermediate_prefill_mask,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def dp_gather_finalize(self, handle: DPGatherHandle) -> ModelRunnerOutput:
        parallel_config = self.vllm_config.parallel_config
        group = self.dp_group
        rank = self.dp_rank
        world = parallel_config.data_parallel_size
        logprobs_per_dp: list = [None] * world

        result = handle.future.result()
        assert isinstance(result, tuple) and len(result) == 2
        send_tensor, logprobs_per_dp = result
        assert isinstance(send_tensor, torch.Tensor)

        my_ids = torch.empty_like(send_tensor[0])
        scatter_list = None
        if rank == 0:
            scatter_list = [send_tensor[i] for i in range(world)]
        dist.scatter(my_ids, scatter_list, src=0, group=group)
        self.dlog("after_results_gather my_ids_shape=%s", tuple(my_ids.shape))

        my_logprobs_val = None
        if handle.any_needs_logprobs:
            my_logprobs: list = [None]
            logprobs_scatter_list = logprobs_per_dp if rank == 0 else None
            dist.scatter_object_list(
                my_logprobs, logprobs_scatter_list, src=0, group=group
            )
            my_logprobs_val = my_logprobs[0]

        if handle.local_has_requests:
            output: ModelRunnerOutput = self.model_executor.collective_rpc(
                "apply_dp_execution_result",
                args=(
                    my_ids,
                    my_logprobs_val,
                    handle.req_ids,
                    handle.req_id_to_index,
                    handle.intermediate_prefill_mask,
                ),
            )[0]
            return output
        return EMPTY_MODEL_RUNNER_OUTPUT

    def _execute_model_dp_gather(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput:
        handle = self.dp_gather_submit(
            scheduler_output, grammar_output, overlap_ok=False
        )
        return self.dp_gather_finalize(handle)

    def _completed_dp_gather_future(self) -> Future[tuple[torch.Tensor, list]]:
        parallel_config = self.vllm_config.parallel_config
        world = parallel_config.data_parallel_size
        batch_size = get_tt_per_lane_max_num_seqs(self.vllm_config)
        return _as_future(
            (torch.zeros((world, batch_size, 1), dtype=torch.int32), [None] * world)
        )


def _as_future(value: _T) -> Future[_T]:
    future: Future[_T] = Future()
    future.set_result(value)
    return future


def _unwrap_single_worker_future(future: Future[list[_T]]) -> Future[_T]:
    single_future: Future[_T] = Future()

    def _set_single_result(done_future: Future[list[_T]]) -> None:
        try:
            results = done_future.result()
            assert len(results) == 1
            single_future.set_result(results[0])
        except Exception as exc:
            single_future.set_exception(exc)

    future.add_done_callback(_set_single_result)
    return single_future
