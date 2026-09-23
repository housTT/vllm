"""Per-request block-table width for uniform vs hybrid (sliding-window) KV configs.

Regression for the Gemma 4 131k prefill failure: with a 1,509-block pool the
runner trimmed the table to 1,509 columns, but the sliding-window groups address
their table by absolute virtual block (``position // 64``), so the model's
sliding-history lookup ran off the end of the table past token 96,576.
"""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_tt_plugin.model_runner import TTModelRunner


def _full():
    return SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=512, dtype=torch.bfloat16)
    )


def _sliding():
    return SimpleNamespace(
        kv_cache_spec=SlidingWindowSpec(
            block_size=64, num_kv_heads=2, head_size=256, dtype=torch.bfloat16, sliding_window=1024
        )
    )


def test_uniform_group_trims_to_pool():
    width = TTModelRunner._max_num_blocks_per_req(131072, 64, 1509, [_full()])
    assert width == 1509


def test_uniform_group_keeps_model_len_when_pool_is_larger():
    width = TTModelRunner._max_num_blocks_per_req(131072, 64, 4096, [_full()])
    assert width == 2048


def test_hybrid_sliding_group_keeps_full_width():
    width = TTModelRunner._max_num_blocks_per_req(131072, 64, 1509, [_full(), _sliding()])
    assert width == 2048
