"""Closed-loop burst through the TT scheduler and upstream's allocator (CPU control).

Sixteen same-length users on a 1,509-block pool (the Gemma 4 p150 layout: 94 blocks per
1,050-token request) fill the pool to the brim and preempt without the reserve; the
per-group default keeps residents growing without a single preemption.
"""

import collections
import unittest

import scheduler_cpu_fixture as F


def _closed_loop(reserve, n_users=16, prompt=1050, out=256, pool_blocks=1509, rounds=4):
    s, _, pool = F.build(0)
    s.waiting = F.g["create_request_queue"](F.Policy.FCFS)
    s.requests = {}
    pool.free = collections.deque(
        F.N(is_null=False, block_id=i, ref_cnt=0) for i in range(1, pool_blocks + 1)
    )
    coordinator = s.kv_cache_manager.coordinator
    coordinator.single_type_managers = coordinator.managers
    s._admission_reserve_blocks_per_seq = reserve

    def launch(rid):
        r = F.req(rid, 0, prompt, F.Status.WAITING)
        r.num_prompt_tokens = prompt
        r.max_tokens = out
        s.waiting.add_request(r)
        s.requests[rid] = r

    launched = 0
    for _ in range(n_users):
        launch(f"u{launched}")
        launched += 1
    preemptions = 0
    peak = 0
    min_free = pool_blocks
    finished = 0
    for _ in range(40000):
        out_ = s.schedule()
        F.drain(s, out_)
        preemptions += len(out_.preempted_req_ids or ())
        peak = max(peak, len(s.running))
        min_free = min(min_free, pool.get_num_free_blocks())
        for rid, r in list(s.requests.items()):
            if r.status == F.Status.RUNNING and r.num_tokens >= r.num_prompt_tokens + r.max_tokens:
                F.finish(s, rid)
                finished += 1
                if launched < rounds * n_users:
                    launch(f"u{launched}")
                    launched += 1
        if not s.requests:
            break
    return dict(preemptions=preemptions, peak=peak, min_free=min_free, finished=finished)


class AdmissionReserveBurstTests(unittest.TestCase):
    def test_without_reserve_the_burst_fills_the_pool_and_preempts(self):
        r = _closed_loop(0)
        self.assertEqual(r["finished"], 64)
        self.assertEqual(r["peak"], 16)
        self.assertEqual(r["min_free"], 0)
        self.assertGreater(r["preemptions"], 0)

    def test_per_group_default_reserve_keeps_growth_headroom(self):
        r = _closed_loop(6 + 2)  # six KV groups in the fixture
        self.assertEqual(r["finished"], 64)
        self.assertLessEqual(r["peak"], 15)
        self.assertGreaterEqual(r["min_free"], 6 * 8)
        self.assertEqual(r["preemptions"], 0)

    def test_thirty_two_users_with_default_reserve(self):
        r = _closed_loop(6 + 2, n_users=32)
        self.assertEqual(r["finished"], 128)
        self.assertEqual(r["preemptions"], 0)

    def test_fixture_pool_restored_after_run(self):
        # The fixture's own tests assume a 4,127-block pool; this module builds its own pools.
        self.assertEqual(F.Pool().get_num_free_blocks(), 4127)


if __name__ == "__main__":
    unittest.main()
