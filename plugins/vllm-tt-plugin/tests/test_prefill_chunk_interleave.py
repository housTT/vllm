# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure scheduler-policy tests for alternating Ornith prefill chunks with decode."""

from types import SimpleNamespace

import pytest

# Keep this first: importing the plugin during vLLM's own bootstrap re-enters a half-built module.
import vllm  # noqa: F401  # isort: skip

from vllm_tt_plugin.scheduler import (
    _PrefillCapPolicy,
    TTScheduler,
    TTSchedulingMode,
    _resolve_chunk_interleave,
    _validate_chunk_interleave_admission,
)


def _config(*, model_type="qwen3_5_moe", chunked=True, partial_prefills=1):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=chunked,
            max_num_partial_prefills=partial_prefills,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )


def _output(tokens=1):
    return SimpleNamespace(total_num_scheduled_tokens=tokens)


def _scheduler(*, running, waiting=True, due=False, interleave=True):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.running = running
    scheduler.waiting = [object()] if waiting else []
    scheduler._forced_mode = TTSchedulingMode.DEFAULT
    scheduler._interleave_prefill_chunks = interleave
    scheduler._decode_due_after_prefill_chunk = due
    return scheduler


def test_interleave_defaults_only_to_the_validated_stateful_model(monkeypatch):
    monkeypatch.delenv("TT_INTERLEAVE_PREFILL_CHUNKS", raising=False)
    assert _resolve_chunk_interleave(_config()) is True
    assert _resolve_chunk_interleave(_config(model_type="gemma4")) is False
    assert _resolve_chunk_interleave(_config(chunked=False)) is False
    with pytest.raises(ValueError, match="max_num_partial_prefills=1"):
        _resolve_chunk_interleave(_config(partial_prefills=2))


def test_interleave_override_is_strict_and_requires_chunking(monkeypatch):
    monkeypatch.setenv("TT_INTERLEAVE_PREFILL_CHUNKS", "off")
    assert _resolve_chunk_interleave(_config()) is False
    monkeypatch.setenv("TT_INTERLEAVE_PREFILL_CHUNKS", "on")
    assert _resolve_chunk_interleave(_config(model_type="gemma4")) is True
    with pytest.raises(ValueError, match="requires enable_chunked_prefill"):
        _resolve_chunk_interleave(_config(chunked=False))
    with pytest.raises(ValueError, match="max_num_partial_prefills=1"):
        _resolve_chunk_interleave(_config(partial_prefills=2))
    monkeypatch.setenv("TT_INTERLEAVE_PREFILL_CHUNKS", "perhaps")
    with pytest.raises(ValueError, match="must be a boolean"):
        _resolve_chunk_interleave(_config())


def test_interleave_requires_exactly_one_prefill_admission_at_every_length():
    _validate_chunk_interleave_admission(
        _PrefillCapPolicy(cap=1, explicit=False, max_prompt_len=1 << 62)
    )
    _validate_chunk_interleave_admission(
        _PrefillCapPolicy(cap=1, explicit=True, max_prompt_len=0)
    )
    for policy in (
        _PrefillCapPolicy(cap=None, explicit=True, max_prompt_len=1 << 62),
        _PrefillCapPolicy(cap=2, explicit=True, max_prompt_len=1 << 62),
        _PrefillCapPolicy(cap=1, explicit=False, max_prompt_len=4096),
    ):
        with pytest.raises(ValueError, match="max_prefills_per_step=1"):
            _validate_chunk_interleave_admission(policy)


def test_a_due_decode_runs_before_the_next_partial_prefill():
    decode = SimpleNamespace(is_prefill_chunk=False)
    partial = SimpleNamespace(is_prefill_chunk=True)
    scheduler = _scheduler(running=[partial, decode], due=True)
    calls = []
    scheduler._schedule_decode_only = lambda: calls.append("decode") or _output()
    scheduler._schedule_prefill_only = lambda: calls.append("prefill") or _output(2048)

    result = TTScheduler.schedule(scheduler)

    assert result.total_num_scheduled_tokens == 1
    assert calls == ["decode"]
    assert scheduler._decode_due_after_prefill_chunk is False


def test_the_first_chunk_marks_decode_due_when_a_prompt_remains_partial():
    decode = SimpleNamespace(is_prefill_chunk=False)
    new_prompt = SimpleNamespace(is_prefill_chunk=False)
    scheduler = _scheduler(running=[decode], due=False)

    def prefill():
        new_prompt.is_prefill_chunk = True
        scheduler.running.append(new_prompt)
        return _output(2048)

    scheduler._schedule_prefill_only = prefill
    scheduler._schedule_decode_only = lambda: pytest.fail("decode is not due before the first chunk")

    TTScheduler.schedule(scheduler)

    assert scheduler._decode_due_after_prefill_chunk is True


def test_a_final_continuation_still_marks_decode_due_before_the_next_prompt():
    partial = SimpleNamespace(is_prefill_chunk=True)
    decode = SimpleNamespace(is_prefill_chunk=False)
    scheduler = _scheduler(running=[partial, decode], due=False)

    def finish_prefill():
        partial.is_prefill_chunk = False
        return _output(128)

    scheduler._schedule_prefill_only = finish_prefill
    scheduler._schedule_decode_only = lambda: pytest.fail("decode is due after, not before, this chunk")

    TTScheduler.schedule(scheduler)

    assert scheduler._decode_due_after_prefill_chunk is True


def test_a_single_chunk_prompt_does_not_change_the_existing_prefill_priority():
    decode = SimpleNamespace(is_prefill_chunk=False)
    scheduler = _scheduler(running=[decode], due=False)
    scheduler._schedule_prefill_only = lambda: _output(128)
    scheduler._schedule_decode_only = lambda: pytest.fail("a short prompt keeps the existing prefill-first policy")

    TTScheduler.schedule(scheduler)

    assert scheduler._decode_due_after_prefill_chunk is False
