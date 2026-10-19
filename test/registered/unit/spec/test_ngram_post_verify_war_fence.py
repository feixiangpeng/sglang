"""The NGRAM post-verify KV move must stay behind the scheduler's WAR fence (#44997).

Black-box mechanism: under the overlap scheduler a request that finishes on a
stop token in batch t is still scheduled into batch t+1's target verify. After
that verify's graph replay published `shared_read_done_event` (the fence the
scheduler's `_apply_war_barrier` waits on before it writes shared buffers), the
NGRAM worker runs `move_accept_tokens_to_target_kvcache` on forward_stream,
which reads `req_to_token[row, seq_len : seq_len + accept]` and copies the
accepted draft KV into those slots. Meanwhile `pop_and_process(batch t)` frees
the finished request's row, a new request takes it, and `alloc_for_extend`
writes the new request's radix-cached prefix slot indices into that row on
schedule_stream. Fenced only by the early event, that write can land before
the move reads the row, and the move then copies stale draft KV over the
shared prefix slots that other live requests read.

The interleaving is reproduced deterministically: a spin kernel on
forward_stream delays the move, so a schedule_stream write fenced by the early
event always lands first. The early fence must show the corruption (that is
what the fix changes); the fence refreshed after the move must not.

Run with:
    python3 -m pytest test/registered/unit/spec/test_ngram_post_verify_war_fence.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner_utils.shared_read_event import (
    refresh_shared_read_done_event,
)
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

ROW = 1  # the finished request's req_pool_idx, reused by the new request
POOL_LEN = 64
SEQ_LEN = 10  # the finished request's committed length when its last verify ran
NUM_DRAFT_TOKENS = 4
NUM_CORRECT_DRAFTS = 2  # accepted drafts; the move covers drafts + bonus = 3 slots
OLD_TARGET_SLOTS = (200, 201, 202)  # the finished request's own over-allocated slots
DRAFT_SLOTS = (400, 401, 402, 403)  # the verify's out_cache_loc for this request
SHARED_PREFIX_BASE = 300  # radix-cached slots the new request's prefix maps to
NEW_PREFIX_LEN = 16  # > SEQ_LEN + 3, so the new row covers the moved range
KV_WIDTH = 8
# Long enough that the fenced schedule_stream write always lands before the move.
SPIN_CYCLES = 500_000_000


class _KvCache:
    """A flat KV store whose move is the one `move_accept_tokens_to_target_kvcache` calls."""

    def __init__(self, device):
        self.kv = torch.arange(512 * KV_WIDTH, device=device, dtype=torch.float32)
        self.kv = self.kv.view(512, KV_WIDTH)

    def move_kv_cache(self, tgt_loc, src_loc):
        self.kv[tgt_loc] = self.kv[src_loc]


def _setup(device):
    req_to_token = torch.zeros((4, POOL_LEN), dtype=torch.int32, device=device)
    req_to_token[ROW, :SEQ_LEN] = torch.arange(100, 100 + SEQ_LEN, device=device)
    req_to_token[ROW, SEQ_LEN : SEQ_LEN + 3] = torch.tensor(
        OLD_TARGET_SLOTS, device=device
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([SEQ_LEN], dtype=torch.int32, device=device),
        req_pool_indices=torch.tensor([ROW], dtype=torch.int32, device=device),
        out_cache_loc=torch.tensor(DRAFT_SLOTS, dtype=torch.int64, device=device),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )
    # drafts 0 and 1 accepted, plus the bonus at index 2
    accept_index = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32, device=device)
    num_correct_drafts = torch.tensor(
        [NUM_CORRECT_DRAFTS], dtype=torch.int32, device=device
    )
    kvcache = _KvCache(device)
    allocator = SimpleNamespace(get_kvcache=lambda: kvcache)
    return batch, accept_index, num_correct_drafts, kvcache, allocator


def _new_request_prefix(device):
    return torch.arange(
        SHARED_PREFIX_BASE,
        SHARED_PREFIX_BASE + NEW_PREFIX_LEN,
        dtype=torch.int32,
        device=device,
    )


def _warm_kernels(device):
    """First calls JIT-compile the triton kernels and grow the allocator cache
    on the host, for longer than the spin runs; done before the race so the
    move is enqueued while the spin is still running."""
    batch, accept_index, num_correct_drafts, _, allocator = _setup(device)
    move_accept_tokens_to_target_kvcache(
        batch, accept_index, num_correct_drafts, allocator
    )
    batch.req_to_token_pool.req_to_token[ROW, :NEW_PREFIX_LEN] = _new_request_prefix(
        device
    )
    torch.cuda.synchronize()


def _run(fence: str):
    """forward_stream: early event, delayed move[, refreshed event];
    schedule_stream: fenced row rewrite for the new request."""
    device = torch.device("cuda")
    _warm_kernels(device)
    forward_stream = torch.cuda.Stream()
    schedule_stream = torch.cuda.Stream()
    batch, accept_index, num_correct_drafts, kvcache, allocator = _setup(device)
    shared_before = kvcache.kv[
        SHARED_PREFIX_BASE + SEQ_LEN : SHARED_PREFIX_BASE + SEQ_LEN + 3
    ].clone()
    draft_kv = kvcache.kv[list(DRAFT_SLOTS[:3])].clone()
    runner = SimpleNamespace(shared_read_done_event=None, device_module=torch.cuda)

    with torch.cuda.stream(forward_stream):
        # What the verify replay publishes (its in-graph metadata reads are done).
        early = torch.cuda.Event()
        early.record()
        runner.shared_read_done_event = early
        torch.cuda._sleep(SPIN_CYCLES)
        move_accept_tokens_to_target_kvcache(
            batch, accept_index, num_correct_drafts, allocator
        )
        if fence == "refreshed":
            refresh_shared_read_done_event(runner)

    # Scheduler._apply_war_barrier, then alloc_for_extend -> write_cache_indices
    # for the new request that took the finished request's row.
    schedule_stream.wait_event(runner.shared_read_done_event)
    with torch.cuda.stream(schedule_stream):
        batch.req_to_token_pool.req_to_token[ROW, :NEW_PREFIX_LEN] = (
            _new_request_prefix(device)
        )
    torch.cuda.synchronize()

    shared_after = kvcache.kv[
        SHARED_PREFIX_BASE + SEQ_LEN : SHARED_PREFIX_BASE + SEQ_LEN + 3
    ]
    own_after = kvcache.kv[list(OLD_TARGET_SLOTS)]
    return {
        "shared_prefix_intact": torch.equal(shared_after, shared_before),
        "own_slots_hold_drafts": torch.equal(own_after, draft_kv),
        "shared_prefix_holds_drafts": torch.equal(shared_after, draft_kv),
    }


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestNgramPostVerifyWarFence(CustomTestCase):
    def test_early_fence_lets_the_row_write_race_the_move(self):
        # The hazard the fix removes: the move reads the new request's row.
        got = _run("early")
        self.assertTrue(got["shared_prefix_holds_drafts"], got)
        self.assertFalse(got["own_slots_hold_drafts"], got)

    def test_refreshed_fence_keeps_the_row_write_behind_the_move(self):
        got = _run("refreshed")
        self.assertTrue(got["shared_prefix_intact"], got)
        self.assertTrue(got["own_slots_hold_drafts"], got)


if __name__ == "__main__":
    unittest.main()
