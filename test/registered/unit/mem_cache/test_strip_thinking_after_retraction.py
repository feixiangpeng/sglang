"""--strip-thinking-cache caps Req.owned_kv_len() at the prompt, so a finished
reasoning request's output is freed instead of inserted. A retracted request is
re-prefilled over prompt + output, and that prefill's checkpoint puts the output
into the tree (cache_protected_len > prompt). Capping below cache_protected_len
then frees slots the tree still owns: they end up in the free list twice and
can be handed to two requests (#41617).

Ported from PR #41767 to main's checkpoint API (checkpoint_kv_cache /
release_kv_cache(checkpoint=...)).
"""

import unittest
from array import array

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req  # noqa: E402
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams  # noqa: E402
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.common import (  # noqa: E402
    checkpoint_kv_cache,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.unified_cache.components.base import (  # noqa: E402
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.runtime_context import get_serving  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

POOL = 128
PROMPT = 8
BEFORE_RETRACT = 6  # output tokens generated before the retraction
AFTER = 3  # decode steps after the re-prefill


def _build(page_size: int):
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=page_size)
    )
    pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kvcache = MHATokenToKVPool(
        size=POOL,
        page_size=page_size,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    if page_size == 1:
        allocator = TokenToKVPoolAllocator(
            size=POOL,
            dtype=torch.float16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
    else:
        allocator = PagedTokenToKVPoolAllocator(
            size=POOL,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )
    return pool, allocator, cache


def _new_req(cache, prompt: int, earlier_output):
    req = Req(
        "r",
        "",
        array("q", range(1, prompt + 1)),
        SamplingParams(temperature=0, max_new_tokens=32),
    )
    req.output_ids = array("q", earlier_output)
    req.reasoning_tokens = len(earlier_output) + 1  # still thinking
    req.last_node = cache.root_node_handle()
    return req


def _prefill_and_decode(pool, allocator, cache, prompt, earlier_output, page_size=1):
    """Prefill prompt + earlier_output (a re-prefill when non-empty), checkpoint
    it as the scheduler does for an unfinished request, then decode AFTER tokens."""
    with get_serving().override(strip_thinking_cache=True):
        return _prefill_and_decode_stripped(
            pool, allocator, cache, prompt, earlier_output, page_size
        )


def _prefill_and_decode_stripped(
    pool, allocator, cache, prompt, earlier_output, page_size
):
    req = _new_req(cache, prompt, earlier_output)
    pool.alloc([req])
    fill = prompt + len(earlier_output)
    req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req.prefix_len = 0
    req.extend_end = fill
    if page_size == 1:
        slots = allocator.alloc(fill)
        pool.write((req.kv.req_pool_idx, slice(0, fill)), slots.to(torch.int32))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = fill
        checkpoint_kv_cache(req, cache)  # batch_result_processor, unfinished
        for _ in range(AFTER):
            slot = allocator.alloc(1)
            pool.write(
                (req.kv.req_pool_idx, req.kv.kv_committed_len), slot.to(torch.int32)
            )
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            req.output_ids.append(99)
    else:
        pages = -(-(fill + AFTER) // page_size)
        slots = allocator.alloc(pages * page_size)
        pool.write((req.kv.req_pool_idx, slice(0, fill)), slots[:fill].to(torch.int32))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = fill
        checkpoint_kv_cache(req, cache)
        for k in range(AFTER):
            pool.write(
                (req.kv.req_pool_idx, req.kv.kv_committed_len),
                slots[fill + k : fill + k + 1].to(torch.int32),
            )
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            req.output_ids.append(99)
    return req


def _release_and_check(tc, pool, allocator, cache, req, *, finish: bool, paged=False):
    tree_slots = set(
        pool.req_to_token[req.kv.req_pool_idx, : req.kv.cache_protected_len].tolist()
    )
    if finish:
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
    with get_serving().override(strip_thinking_cache=True):
        release_kv_cache(req, cache, checkpoint=finish)

    if not paged:
        free = allocator.free_pages.tolist()
        tc.assertEqual(sorted(tree_slots & set(free)), [], "tree-owned slots freed")
    cache.evict(EvictParams(num_tokens=POOL))
    if paged:
        allocator.merge_and_sort_free()
    free = allocator.free_pages.tolist()
    tc.assertEqual(len(free), len(set(free)), "a KV slot is free twice")
    tc.assertEqual(allocator.available_size(), POOL)


class TestStripThinkingAfterRetraction(CustomTestCase):
    def setUp(self):
        self.pool, self.allocator, self.cache = _build(page_size=1)

    def _run(self, earlier_output):
        return _prefill_and_decode(
            self.pool, self.allocator, self.cache, PROMPT, earlier_output
        )

    def test_finish_after_reprefill_frees_each_slot_once(self):
        req = self._run(list(range(100, 100 + BEFORE_RETRACT)))
        _release_and_check(
            self, self.pool, self.allocator, self.cache, req, finish=True
        )

    def test_second_retraction_frees_each_slot_once(self):
        req = self._run(list(range(100, 100 + BEFORE_RETRACT)))
        _release_and_check(
            self, self.pool, self.allocator, self.cache, req, finish=False
        )

    def test_thinking_is_not_cached_after_a_retraction(self):
        # The flag's contract: on finish the tree keeps the prompt only.
        req = self._run(list(range(100, 100 + BEFORE_RETRACT)))
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        with get_serving().override(strip_thinking_cache=True):
            release_kv_cache(req, self.cache, checkpoint=True)
        self.assertLessEqual(self.cache.evictable_size(), PROMPT)

    def test_without_retraction_output_is_still_stripped(self):
        req = self._run([])
        with get_serving().override(strip_thinking_cache=True):
            self.assertEqual(req.owned_kv_len(), PROMPT)
        _release_and_check(
            self, self.pool, self.allocator, self.cache, req, finish=True
        )


class TestStripThinkingAfterRetractionPaged(CustomTestCase):
    """page_size=4. With an unaligned prompt the tree-owned page is freed by the
    unaligned-tail free instead of the over-allocated range; the same bound on
    owned_kv_len() covers both."""

    PAGE = 4

    def test_reprefill_release_frees_each_page_once(self):
        for prompt in (8, 10):  # aligned, unaligned
            for finish in (True, False):
                with self.subTest(prompt=prompt, finish=finish):
                    pool, allocator, cache = _build(page_size=self.PAGE)
                    req = _prefill_and_decode(
                        pool,
                        allocator,
                        cache,
                        prompt,
                        list(range(100, 100 + BEFORE_RETRACT)),
                        page_size=self.PAGE,
                    )
                    _release_and_check(
                        self, pool, allocator, cache, req, finish=finish, paged=True
                    )


if __name__ == "__main__":
    unittest.main()
