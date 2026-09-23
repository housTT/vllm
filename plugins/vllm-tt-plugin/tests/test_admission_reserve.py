"""KV admission reserve: hold new prompts back so residents keep room to grow.

Uses upstream's real ``KVCacheManager`` with the Gemma 4 hybrid layout (one
full-attention group with 128-token blocks, five sliding-window groups with
64-token blocks and a 1,024 window) so the per-request block arithmetic is the
production one: a 1,050-token prompt costs 94 blocks.
"""

import inspect
from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_tt_plugin.scheduler import TTScheduler, _admission_reserve_blocks_per_seq


def _groups():
    full = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=512, dtype=torch.bfloat16)
    sliding = SlidingWindowSpec(
        block_size=64, num_kv_heads=2, head_size=256, dtype=torch.bfloat16, sliding_window=1024
    )
    groups = [KVCacheGroupSpec(layer_names=[f"f{i}" for i in range(5)], kv_cache_spec=full)]
    for g in range(5):
        groups.append(KVCacheGroupSpec(layer_names=[f"s{g}_{i}" for i in range(5)], kv_cache_spec=sliding))
    return groups


def _manager(num_blocks, max_model_len=131072):
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=_groups())
    kwargs = dict(kv_cache_config=cfg, max_model_len=max_model_len, enable_caching=False, hash_block_size=64)
    allowed = inspect.signature(KVCacheManager.__init__).parameters
    return KVCacheManager(**{k: v for k, v in kwargs.items() if k in allowed})


def _request(rid, prompt_tokens, max_tokens=256):
    allowed = inspect.signature(Request.__init__).parameters
    kwargs = dict(
        request_id=rid,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        eos_token_id=None,
    )
    for name in ("multi_modal_kwargs", "multi_modal_hashes", "multi_modal_placeholders", "mm_features",
                 "lora_request", "block_hasher"):
        if name in allowed:
            kwargs[name] = None
    if "arrival_time" in allowed:
        kwargs["arrival_time"] = 0.0
    return Request(**kwargs)


def _scheduler_stub(manager, per_seq):
    stub = TTScheduler.__new__(TTScheduler)
    stub.kv_cache_manager = manager
    stub._admission_reserve_blocks_per_seq = per_seq
    return stub


def test_reserve_holds_back_the_prompts_that_would_fill_the_pool():
    manager = _manager(1509)
    waiting = [_request(f"r{i}", 1050) for i in range(32)]
    # Without the reserve upstream admits 16 x 94 = 1,504 of 1,509 blocks.
    blocked = _scheduler_stub(manager, 0)._admission_blocked_ids(waiting, 0)
    assert blocked == set()
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids(waiting, 0)
    admitted = 32 - len(blocked)
    # 15 x 94 = 1,410 blocks admitted leaves 99 >= 4 x 15 free; a 16th would leave 5.
    assert admitted == 15
    assert blocked == {f"r{i}" for i in range(15, 32)}


def test_reserve_counts_residents_already_decoding():
    manager = _manager(1509)
    residents = [_request(f"d{i}", 1050) for i in range(10)]
    for r in residents:
        assert manager.allocate_slots(r, 1050) is not None
    waiting = [_request(f"w{i}", 1050) for i in range(8)]
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids(waiting, len(residents))
    # 569 blocks free; each admit needs 94 + 4 more reserve: 5 fit (569 - 470 = 99 >= 4 x 15), a 6th does not.
    assert 8 - len(blocked) == 5


def test_footprint_is_the_steady_state_not_the_whole_prompt():
    stub = _scheduler_stub(_manager(1509), 4)
    # 1,050 tokens: 9 full-layer blocks (128) + 5 x 17 sliding blocks.
    assert stub._resident_footprint_blocks(1050) == 9 + 5 * 17
    # 26,698 tokens (opencode turn 6): the sliding groups stay at the window.
    assert stub._resident_footprint_blocks(26698) == 209 + 5 * 17
    # One request at max_model_len fits the 1,509-block pool.
    assert stub._resident_footprint_blocks(131072) == 1024 + 5 * 17


def test_long_prompts_are_admitted_on_an_empty_pool_and_gated_by_footprint():
    manager = _manager(1509)
    waiting = [_request("t6", 26698), _request("t7", 35829), _request("ctx", 131072)]
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids(waiting, 0)
    # 294 + 365 blocks fit; the 131k request (1,109) does not fit behind them.
    assert blocked == {"ctx"}
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids([_request("ctx", 131072)], 0)
    assert blocked == set()


def test_queue_order_is_respected():
    manager = _manager(300)
    waiting = [_request("long", 4150), _request("short", 100)]
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids(waiting, 0)
    # The long prompt (118 blocks) fits a 300-block pool; both are admitted in order.
    assert blocked == set()
    waiting = [_request("a", 16400), _request("b", 100)]
    blocked = _scheduler_stub(manager, 4)._admission_blocked_ids(waiting, 0)
    # 129 + 85 = 214 blocks for "a"; "b" needs 6 more with reserve 8: admitted behind it.
    assert blocked == set()


def test_reserve_knob_defaults_and_reads_tt_config():
    assert _admission_reserve_blocks_per_seq(None) is None
    cfg = SimpleNamespace(additional_config={"tt": {"kv_admission_reserve_blocks_per_seq": 0}})
    assert _admission_reserve_blocks_per_seq(cfg) == 0
    cfg = SimpleNamespace(additional_config={"tt": {"kv_admission_reserve_blocks_per_seq": 8}})
    assert _admission_reserve_blocks_per_seq(cfg) == 8


def test_default_reserve_is_one_block_per_group_plus_two():
    manager = _manager(1509)
    stub = TTScheduler.__new__(TTScheduler)
    stub.kv_cache_manager = manager
    stub.vllm_config = SimpleNamespace(additional_config={})
    waiting = [_request(f"r{i}", 1050) for i in range(32)]
    blocked = stub._admission_blocked_ids(waiting, 0)
    # 6 groups -> 8 blocks per resident: 14 x 94 = 1,316 admitted leaves 193 >= 8 x 14; a 15th would leave 99 < 120.
    assert stub._admission_reserve_blocks_per_seq == 8
    assert 32 - len(blocked) == 14
