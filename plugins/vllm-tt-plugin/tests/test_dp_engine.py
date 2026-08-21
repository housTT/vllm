# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for gathered-DP mode negotiation."""

from types import SimpleNamespace

import pytest
import vllm_tt_plugin.engine as engine_module
from vllm_tt_plugin.engine import (
    TTDPEngineCoreProc,
    TTEngineCoreProc,
    _get_grouped_ornith_prefill_target,
    _get_input_queue_batching_delay,
)
from vllm_tt_plugin.platform import _set_tt_engine_core_proc_classes
from vllm_tt_plugin.scheduler import TTSchedulingMode

from vllm.v1.engine import EngineCoreRequestType


def _core_with_scheduler(scheduler):
    core = TTDPEngineCoreProc.__new__(TTDPEngineCoreProc)
    core.scheduler = scheduler
    core.dp_group = object()
    core.dlog = lambda *args, **kwargs: None
    return core


def _queue_delay_config(
    *,
    model_type: str,
    max_num_seqs: int,
    max_num_partial_prefills: int,
    override: float | None = None,
):
    tt_config = {}
    if override is not None:
        tt_config["input_queue_batching_delay"] = override
    return SimpleNamespace(
        additional_config={"tt": tt_config},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_partial_prefills=max_num_partial_prefills,
        ),
    )


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class _ScriptedInputQueue:
    """Deterministic blocking queue without an ``empty()`` race surface."""

    def __init__(self, clock, events=()):
        self.clock = clock
        self.events = list(events)
        self.blocking_timeouts = []
        self.nowait_calls = 0

    def empty(self):
        return not self.events or self.events[0][0] > 0

    def get(self, *, timeout=None):
        self.blocking_timeouts.append(timeout)
        if not self.events:
            if timeout is None:
                raise AssertionError("unexpected indefinite queue wait")
            self.clock.advance(timeout)
            raise engine_module.queue.Empty
        arrival_delay, request = self.events[0]
        if timeout is not None and arrival_delay > timeout:
            self.clock.advance(timeout)
            self.events[0] = (arrival_delay - timeout, request)
            raise engine_module.queue.Empty
        self.events.pop(0)
        self.clock.advance(arrival_delay)
        return request

    def get_nowait(self):
        self.nowait_calls += 1
        if not self.events or self.events[0][0] > 0:
            raise engine_module.queue.Empty
        _, request = self.events.pop(0)
        return request


class _QueueScheduler:
    def __init__(self, *, num_running=0, num_waiting=0):
        self.num_running = num_running
        self.num_waiting = num_waiting

    def has_requests(self):
        return self.num_running > 0 or self.num_waiting > 0

    def get_request_counts(self):
        return self.num_running, self.num_waiting


def _queue_processing_core(
    config,
    scheduler,
    input_queue,
    *,
    engines_running=False,
    core_cls=TTDPEngineCoreProc,
):
    core = core_cls.__new__(core_cls)
    core.vllm_config = config
    core.scheduler = scheduler
    core.input_queue = input_queue
    core.aborts_queue = engine_module.queue.Queue()
    core.engines_running = engines_running
    core.batch_queue = None
    core._dp_in_flight = None
    core._scheduler_paused = False
    handled = []

    def handle(request_type, request):
        handled.append((request_type, request))
        if request_type == "add":
            scheduler.num_waiting += 1

    core._handle_client_request = handle
    return core, handled


def test_single_engine_blocks_when_idle_then_coalesces_to_grouped_target(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.020, ("add", "request-1")),
            (0.050, ("add", "request-2")),
            (0.050, ("add", "request-3")),
            (0.050, ("add", "request-4")),
        ],
    )
    scheduler = _QueueScheduler()
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        engines_running=False,
        core_cls=TTEngineCoreProc,
    )
    core.aborts_queue.put_nowait(["stale-abort"])
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [
        ("add", "request-1"),
        ("add", "request-2"),
        ("add", "request-3"),
        ("add", "request-4"),
    ]
    assert scheduler.num_waiting == 4
    assert input_queue.blocking_timeouts[0] is None
    assert input_queue.blocking_timeouts[1:] == pytest.approx([0.250, 0.200, 0.150])
    assert input_queue.nowait_calls == 1
    assert core.aborts_queue.empty()
    assert clock.now == pytest.approx(0.170)


def test_single_engine_controls_before_first_add_do_not_consume_deadline(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.100, ("utility", "status")),
            (0.100, ("add", "request-1")),
            (0.200, ("add", "request-2")),
        ],
    )
    scheduler = _QueueScheduler()
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        engines_running=False,
        core_cls=TTEngineCoreProc,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [
        ("utility", "status"),
        ("add", "request-1"),
        ("add", "request-2"),
    ]
    assert input_queue.blocking_timeouts[:2] == [None, None]
    assert input_queue.blocking_timeouts[2:] == pytest.approx([0.250, 0.050])
    # The 250 ms deadline starts at t=0.200 after the ADD, not at t=0.100
    # after the control message.
    assert clock.now == pytest.approx(0.450)


def test_abort_that_empties_waiting_queue_stops_blocking(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [(0.050, ("abort", "request-1"))],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        core_cls=TTEngineCoreProc,
    )

    def handle(request_type, request):
        handled.append((request_type, request))
        scheduler.num_waiting = 0

    core._handle_client_request = handle
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("abort", "request-1")]
    assert input_queue.blocking_timeouts == pytest.approx([0.250])
    assert input_queue.nowait_calls == 1
    assert clock.now == pytest.approx(0.050)


def test_executor_failure_during_coalescing_propagates(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [(0.0, (EngineCoreRequestType.EXECUTOR_FAILED, b""))],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, _ = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        core_cls=TTEngineCoreProc,
    )
    core._handle_client_request = TTEngineCoreProc._handle_client_request.__get__(
        core, TTEngineCoreProc
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    with pytest.raises(RuntimeError, match="Executor failed"):
        core._process_input_queue()

    assert input_queue.blocking_timeouts == pytest.approx([0.250])


def test_single_engine_active_work_drains_without_coalescing(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.0, ("abort", "running-request")),
            (0.010, ("add", "future-request")),
        ],
    )
    scheduler = _QueueScheduler(num_running=1, num_waiting=1)
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        core_cls=TTEngineCoreProc,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("abort", "running-request")]
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 2
    assert len(input_queue.events) == 1
    assert clock.now == 0.0


@pytest.mark.parametrize(
    ("model_type", "max_num_seqs", "max_num_partial_prefills"),
    [
        ("gemma4", 8, 4),
        ("qwen3_5_moe", 1, 4),
    ],
)
def test_single_engine_non_grouped_config_preserves_base_queue_behavior(
    monkeypatch,
    model_type,
    max_num_seqs,
    max_num_partial_prefills,
):
    config = _queue_delay_config(
        model_type=model_type,
        max_num_seqs=max_num_seqs,
        max_num_partial_prefills=max_num_partial_prefills,
    )
    core = TTEngineCoreProc.__new__(TTEngineCoreProc)
    core.vllm_config = config
    delegated = []

    def base_process_input_queue(instance):
        delegated.append(instance)

    monkeypatch.setattr(
        engine_module.EngineCoreProc,
        "_process_input_queue",
        base_process_input_queue,
    )

    core._process_input_queue()

    assert delegated == [core]


def test_single_engine_explicit_delay_opts_other_model_into_coalescing(monkeypatch):
    config = _queue_delay_config(
        model_type="gemma4",
        max_num_seqs=8,
        max_num_partial_prefills=4,
        override=0.013,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock)
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        core_cls=TTEngineCoreProc,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == pytest.approx([0.013])
    assert clock.now == pytest.approx(0.013)


def test_platform_routes_single_and_gathered_engines_through_tt_queue_handling():
    parallel_config = SimpleNamespace(
        engine_core_proc_cls="vllm.v1.engine.core.EngineCoreProc",
        dp_engine_core_proc_cls="vllm.v1.engine.core.DPEngineCoreProc",
    )

    _set_tt_engine_core_proc_classes(parallel_config)

    assert (
        parallel_config.engine_core_proc_cls == "vllm_tt_plugin.engine.TTEngineCoreProc"
    )
    assert (
        parallel_config.dp_engine_core_proc_cls
        == "vllm_tt_plugin.engine.TTDPEngineCoreProc"
    )


@pytest.mark.parametrize(
    ("model_type", "max_num_seqs", "max_num_partial_prefills", "expected"),
    [
        ("qwen3_5_moe", 8, 4, 0.250),
        ("qwen3_5_moe", 1, 4, 0.002),
        ("qwen3_5_moe", 8, 1, 0.002),
        ("gemma4", 8, 4, 0.002),
    ],
)
def test_input_queue_batching_delay_is_wider_only_for_grouped_ornith_prefill(
    model_type,
    max_num_seqs,
    max_num_partial_prefills,
    expected,
):
    config = _queue_delay_config(
        model_type=model_type,
        max_num_seqs=max_num_seqs,
        max_num_partial_prefills=max_num_partial_prefills,
    )

    assert _get_input_queue_batching_delay(config) == expected


@pytest.mark.parametrize("override", [0.0, 0.013])
def test_input_queue_batching_delay_honors_explicit_tt_override(override):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
        override=override,
    )

    assert _get_input_queue_batching_delay(config) == override


@pytest.mark.parametrize(
    ("max_num_seqs", "max_num_partial_prefills", "expected"),
    [(8, 4, 4), (2, 4, 2), (8, 2, 2)],
)
def test_grouped_ornith_prefill_target_uses_tighter_scheduler_limit(
    max_num_seqs,
    max_num_partial_prefills,
    expected,
):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=max_num_seqs,
        max_num_partial_prefills=max_num_partial_prefills,
    )

    assert _get_grouped_ornith_prefill_target(config) == expected


def test_input_queue_blocking_get_wakes_immediately_and_stops_blocking_at_target(
    monkeypatch,
):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock, [(0.0, ("add", "request-4"))])
    scheduler = _QueueScheduler(num_waiting=3)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("add", "request-4")]
    assert scheduler.num_waiting == 4
    assert input_queue.blocking_timeouts == pytest.approx([0.250])
    assert input_queue.nowait_calls == 1
    assert clock.now == 0.0


def test_input_queue_does_not_block_past_grouped_target(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock, [(0.010, ("add", "request-5"))])
    scheduler = _QueueScheduler(num_waiting=4)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 1
    assert len(input_queue.events) == 1


def test_input_queue_drains_arrived_control_message_at_grouped_target(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock, [(0.0, ("abort", "request-2"))])
    scheduler = _QueueScheduler(num_waiting=4)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("abort", "request-2")]
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 2


def test_active_wave_rank_without_local_requests_never_coalesces(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock)
    scheduler = _QueueScheduler()
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        engines_running=True,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 1
    assert clock.now == 0.0


def test_active_wave_with_waiting_request_never_coalesces(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [(0.010, ("add", "future-request"))],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        engines_running=True,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 1
    assert len(input_queue.events) == 1
    assert clock.now == 0.0


def test_running_decode_never_coalesces_but_drains_arrived_messages(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.0, ("abort", "running-request")),
            (0.010, ("add", "future-request")),
        ],
    )
    scheduler = _QueueScheduler(num_running=1, num_waiting=1)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("abort", "running-request")]
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 2
    assert len(input_queue.events) == 1
    assert clock.now == 0.0


def test_zero_delay_override_drains_arrived_messages_without_blocking(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
        override=0.0,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.0, ("add", "request-2")),
            (0.010, ("add", "future-request")),
        ],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("add", "request-2")]
    assert scheduler.num_waiting == 2
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 2
    assert len(input_queue.events) == 1
    assert clock.now == 0.0


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("batch_queue", [object()]),
        ("_dp_in_flight", object()),
        ("_scheduler_paused", True),
    ],
)
def test_input_queue_never_coalesces_while_other_engine_work_is_pending(
    monkeypatch,
    attribute,
    value,
):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock)
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    setattr(core, attribute, value)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == []
    assert input_queue.nowait_calls == 1
    assert clock.now == 0.0


def test_input_queue_uses_one_absolute_coalescing_deadline(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.100, ("utility", "status")),
            (0.100, ("add", "request-2")),
        ],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(config, scheduler, input_queue)
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("utility", "status"), ("add", "request-2")]
    assert input_queue.blocking_timeouts == pytest.approx([0.250, 0.150, 0.050])
    assert input_queue.nowait_calls == 1
    assert clock.now == pytest.approx(0.250)


def test_input_queue_drains_fifo_messages_already_queued_at_deadline(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(
        clock,
        [
            (0.0, ("utility", "slow-status")),
            (0.0, ("abort", "request-2")),
        ],
    )
    scheduler = _QueueScheduler(num_waiting=1)
    core, handled = _queue_processing_core(config, scheduler, input_queue)

    def handle(request_type, request):
        handled.append((request_type, request))
        if request == "slow-status":
            clock.advance(0.250)

    core._handle_client_request = handle
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == [("utility", "slow-status"), ("abort", "request-2")]
    assert input_queue.blocking_timeouts == pytest.approx([0.250])
    assert input_queue.nowait_calls == 2
    assert clock.now == pytest.approx(0.250)


def test_idle_input_queue_timeout_returns_to_dp_loop_once(monkeypatch):
    config = _queue_delay_config(
        model_type="qwen3_5_moe",
        max_num_seqs=8,
        max_num_partial_prefills=4,
    )
    clock = _FakeClock()
    input_queue = _ScriptedInputQueue(clock)
    scheduler = _QueueScheduler()
    core, handled = _queue_processing_core(
        config,
        scheduler,
        input_queue,
        engines_running=False,
    )
    monkeypatch.setattr(engine_module.time, "monotonic", clock.monotonic)

    core._process_input_queue()

    assert handled == []
    assert input_queue.blocking_timeouts == pytest.approx([0.002])
    assert clock.now == pytest.approx(0.002)


def test_dp_negotiation_prefers_running_prefill_continuation(monkeypatch):
    scheduler = SimpleNamespace(
        waiting=[],
        running=[SimpleNamespace(is_prefill_chunk=True)],
        max_num_running_reqs=1,
    )
    core = _core_with_scheduler(scheduler)

    def all_reduce(tensor, *, op, group):
        assert tensor.tolist() == [1]
        assert op == engine_module.dist.ReduceOp.MAX
        assert group is core.dp_group

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)

    assert core._dp_negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY
    assert core._dp_gather_forced_mode == TTSchedulingMode.PREFILL_ONLY
