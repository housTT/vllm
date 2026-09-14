# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-extracted CPU control of TT scheduler and hybrid KV allocation.

No vLLM, torch or TT modules are imported. Only the device completion boundary,
request/config objects, and block pool are substitutes. The block pool tracks
individual ownership, includes the reserved null block, and has 4128 entries.
The production allocator supplies sliding eviction and allocation arithmetic.
"""

import ast
import collections
import contextlib
import dataclasses
import enum
import functools
import hashlib
import heapq
import pathlib
import time
import typing
from types import SimpleNamespace as N

ROOT = pathlib.Path(__file__).resolve().parents[3]
files = {}
g = {
    "__name__": "__main__",
    "ClassVar": typing.ClassVar,
    "dataclass": dataclasses.dataclass,
    "Enum": enum.Enum,
    "time": time,
    "record_function_or_nullcontext": lambda _: contextlib.nullcontext(),
    "cdiv": lambda x, y: (x + y - 1) // y,
}


def extract(path, name, methods=None, bases=None):
    path = ROOT / path
    src = path.read_text()
    files[str(path)] = hashlib.sha256(src.encode()).hexdigest()
    c = next(
        x for x in ast.parse(src).body if isinstance(x, ast.ClassDef) and x.name == name
    )
    if methods is not None:
        c.body = [
            x
            for x in c.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
            and x.name in methods
        ]
    if bases is not None:
        c.bases = [ast.Name(x, ast.Load()) for x in bases]
    mod = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), c], []
    )
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), g)
    return g[name]


class Status(enum.Enum):
    RUNNING = 1
    WAITING = 2
    PREEMPTED = 3
    WAITING_FOR_REMOTE_KVS = 4
    WAITING_FOR_FSM = 5
    WAITING_FOR_STREAMING_REQ = 6


class Policy(enum.Enum):
    FCFS = 1
    PRIORITY = 2


class Blocks:
    def __init__(self, bs):
        self.blocks = tuple(bs)

    def get_block_ids(self, allow_none=False):
        return tuple([b.block_id for b in row] for row in self.blocks)


class Pool:
    def __init__(self):
        self.null_block = N(is_null=True, block_id=0, ref_cnt=0)
        self.free = collections.deque(
            N(is_null=False, block_id=i, ref_cnt=0) for i in range(1, 4128)
        )

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, n):
        assert n <= len(self.free)
        bs = [self.free.popleft() for _ in range(n)]
        for b in bs:
            b.ref_cnt = 1
        return bs

    def free_blocks(self, bs):
        for b in bs:
            if not b.is_null:
                assert b.ref_cnt == 1
                b.ref_cnt = 0
                self.free.append(b)


g.update(
    RequestStatus=Status,
    SchedulingPolicy=Policy,
    NewRequestData=N(from_request=lambda r, *args: N(req_id=r.request_id)),
    EngineCoreEventType=N(PREEMPTED="PREEMPTED", SCHEDULED="SCHEDULED"),
)
single = "vllm/v1/core/single_type_kv_cache_manager.py"
Single = extract(
    single,
    "SingleTypeKVCacheManager",
    [
        "get_num_blocks_to_allocate",
        "_get_num_evictable_blocks",
        "allocate_new_blocks",
        "remove_skipped_blocks",
        "free",
        "get_num_skipped_tokens",
    ],
    [],
)
Slide = extract(
    single,
    "SlidingWindowManager",
    ["get_num_skipped_tokens"],
    ["SingleTypeKVCacheManager"],
)


class Coordinator:
    def __init__(self, pool):
        self.managers = []
        for index in range(6):
            m = object.__new__(Slide if index < 5 else Single)
            m.block_size = 64 if index < 5 else 128
            m.sliding_window = 1024
            m.req_to_blocks = collections.defaultdict(list)
            m.num_cached_block = {}
            m._null_block = pool.null_block
            m.block_pool = pool
            self.managers.append(m)

    def remove_skipped_blocks(self, rid, computed):
        for m in self.managers:
            m.remove_skipped_blocks(rid, computed)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_tokens_main_model,
        **kw,
    ):
        return sum(
            m.get_num_blocks_to_allocate(
                request_id, num_tokens, b, total_computed_tokens, num_tokens_main_model
            )
            for m, b in zip(self.managers, new_computed_blocks)
        )

    def allocate_new_blocks(self, rid, n, nmain, *_):
        return tuple(m.allocate_new_blocks(rid, n, nmain) for m in self.managers)


KV = extract(
    "vllm/v1/core/kv_cache_manager.py", "KVCacheManager", ["allocate_slots"], []
)
Scheduler = extract(
    "vllm/v1/core/sched/scheduler.py",
    "Scheduler",
    ["schedule", "_preempt_request", "_update_after_schedule"],
    [],
)
Async = extract(
    "vllm/v1/core/sched/async_scheduler.py",
    "AsyncScheduler",
    ["_update_after_schedule"],
    ["Scheduler"],
)
tt = "plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py"
extract(tt, "_PendingOutputs")
extract(tt, "TTSchedulingMode")
module = ast.parse((ROOT / tt).read_text())
helpers = [node for node in module.body if isinstance(node, ast.FunctionDef)]
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *helpers],
                [],
            )
        ),
        str(ROOT / tt),
        "exec",
    ),
    g,
)
TT = extract(tt, "TTScheduler", bases=["AsyncScheduler"])


class Request(N):
    @property
    def num_output_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    __hash__ = object.__hash__

    def __lt__(self, other):
        return (self.priority, self.arrival_time, self.request_id) < (
            other.priority,
            other.arrival_time,
            other.request_id,
        )

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)

    def record_event(self, event, ts):
        self.events.append(event)


def req(rid, computed, ntokens, status):
    return Request(
        request_id=rid,
        num_computed_tokens=computed,
        num_tokens=ntokens,
        num_prompt_tokens=8192,
        max_tokens=128,
        num_output_placeholders=0,
        spec_token_ids=[],
        has_encoder_inputs=False,
        is_prefill_chunk=False,
        status=status,
        num_preemptions=0,
        num_cached_tokens=-1,
        use_structured_output=False,
        events=[],
        priority=0,
        arrival_time=0,
        discard_latest_async_tokens=False,
    )


def build(n):
    pool = Pool()
    c = Coordinator(pool)
    kv = object.__new__(KV)
    kv.max_model_len = 262144
    kv.coordinator = c
    kv.block_pool = pool
    kv.empty_kv_cache_blocks = Blocks(tuple(() for _ in range(6)))
    kv.enable_caching = False
    kv.create_kv_cache_blocks = Blocks
    kv.new_step_starts = lambda: None
    kv.get_computed_blocks = lambda r: (kv.empty_kv_cache_blocks, 0)
    kv.get_blocks = lambda rid: Blocks(tuple(m.req_to_blocks[rid] for m in c.managers))
    kv.get_num_common_prefix_blocks = lambda _: [0] * 6
    kv.free = lambda r: [m.free(r.request_id) for m in c.managers]
    s = object.__new__(TT)
    s.running = [req("d" + str(i), 8194, 8195, Status.RUNNING) for i in range(n)]
    for r in s.running:
        for m in c.managers:
            if m.block_size == 64:
                m.req_to_blocks[r.request_id] = [
                    pool.null_block
                ] * 112 + pool.get_new_blocks(17)
            else:
                m.req_to_blocks[r.request_id] = pool.get_new_blocks(65)
    w = req("waiting", 0, 8192, Status.WAITING)
    s.waiting = g["create_request_queue"](
        s.policy if hasattr(s, "policy") else Policy.FCFS
    )
    s.waiting.add_request(w)
    s.requests = {r.request_id: r for r in [*s.running, w]}
    s._forced_mode = g["TTSchedulingMode"].DEFAULT
    s._decode_after_empty_prefill = False
    s.max_num_running_reqs = 32
    s.max_num_scheduled_tokens = 2048
    s.max_num_encoder_input_tokens = 0
    s.max_model_len = 262144
    s.scheduler_config = N(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        async_scheduling=True,
    )
    s.kv_cache_manager = kv
    s.encoder_cache_manager = N(free=lambda r: None, get_freed_mm_hashes=lambda: [])
    s.need_mamba_block_aligned_split = False
    s.num_lookahead_tokens = 0
    s.use_eagle = False
    s.policy = Policy.FCFS
    s.lora_config = None
    s.connector = None
    s.ec_connector = None
    s.log_stats = True
    s.use_v2_model_runner = False
    s.kv_cache_config = N(kv_cache_groups=[None] * 6)
    s.prev_step_scheduled_req_ids = set()
    s.finished_req_ids = set()
    s.is_encoder_decoder = False
    s.num_spec_tokens = 0
    s._spec_token_placeholders = []
    s._make_cached_request_data = lambda *args: g["CachedRequestData"].make_empty()
    return s, w, pool


def drain(s, out):
    for rid in out.num_scheduled_tokens:
        r = s.requests[rid]
        if not r.is_prefill_chunk:
            r.num_tokens += 1
            r.num_output_placeholders -= 1
            if s.scheduler_config.async_scheduling:
                assert not g["_PendingOutputs"].for_request(r).is_next_stale()


def step(s):
    out = s.schedule()
    drain(s, out)
    return out


def finish(s, rid):
    r = s.requests.pop(rid)
    if r in s.running:
        s.running.remove(r)
    else:
        s.waiting.remove_requests([r])
    s.kv_cache_manager.free(r)
    s.finished_req_ids.add(rid)


# Use the production FCFS and priority queues for deferral ordering checks.
g.update(heapq=heapq, deque=collections.deque, abstractmethod=lambda f: f)
queue_path = "vllm/v1/core/sched/request_queue.py"
extract(queue_path, "RequestQueue", bases=[])
extract(queue_path, "FCFSRequestQueue", bases=["deque", "RequestQueue"])
extract(queue_path, "PriorityRequestQueue", bases=["RequestQueue"])
g["create_request_queue"] = lambda policy: (
    g["PriorityRequestQueue"]()
    if policy == Policy.PRIORITY
    else g["FCFSRequestQueue"]()
)

g.update(bc_linter_include=lambda c: c, cached_property=functools.cached_property)
output_path = "vllm/v1/core/sched/output.py"
extract(output_path, "CachedRequestData")
extract(output_path, "SchedulerOutput")

Mode = g["TTSchedulingMode"]


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def item(self):
        return self.values[0]

    def tolist(self):
        return list(self.values)

    def copy_(self, other):
        self.values = other.tolist()


g.update(
    torch=N(tensor=lambda values, **kw: Tensor(values), int32="int32"),
    dist=N(ReduceOp=N(MAX="MAX", SUM="SUM"), all_reduce=lambda *args, **kw: None),
)
DP = extract(
    "plugins/vllm-tt-plugin/src/vllm_tt_plugin/engine.py",
    "TTDPEngineCoreProc",
    [
        "_dp_negotiate_forced_mode",
        "_dp_apply_forced_mode",
        "_dp_schedule_with_zero_prefill_fallback",
        "step",
        "step_dp_with_batch_queue",
    ],
    bases=[],
)


def dp_core(s):
    core = object.__new__(DP)
    core.scheduler = s
    core.dp_group = None
    core.dlog = lambda *a, **kw: None
    s.has_unfinished_requests = lambda: bool(s.running or s.waiting)
    return core


lane_path = "plugins/vllm-tt-plugin/src/vllm_tt_plugin/lane_scheduler.py"
extract(lane_path, "_LaneStepState")
module = ast.parse((ROOT / lane_path).read_text())
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "merge_lane_scheduler_outputs",
        "_set_tt_step_state",
        "_get_tt_step_state",
    ):
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        [
                            ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                            node,
                        ],
                        [],
                    )
                ),
                str(ROOT / lane_path),
                "exec",
            ),
            g,
        )
g["_TT_STEP_STATE_ATTR"] = "_tt_step_state"
Lane = extract(
    lane_path,
    "TTLaneCoordinator",
    [
        "_local_prefill_intent",
        "_negotiate_forced_mode",
        "_schedule_all_lanes",
        "schedule",
    ],
    bases=[],
)


def lane_coordinator(schedulers):
    lane = object.__new__(Lane)
    lane.lanes = schedulers
    lane._per_lane_max = 32
    lane._build_step_plan = lambda outputs, merged, is_decode: N(is_decode=is_decode)
    return lane
