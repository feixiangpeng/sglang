"""
Regression for sgl-project/sglang#38452: with a storage backend, a host KV pool that is
not larger than the device pool cannot stage any L3 hit once the device pool is full
(write-through keeps a host copy of every device-resident token and host eviction frees
only host-only entries). The server now warns at storage attach and logs the dropped
hits. These tests cover the decision helpers and the attach-time warning.

CPU only. Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_storage_staging_headroom.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.unified_cache.storage_attachment import (
    StorageAttachment,
    should_log_storage_hit_drop,
    storage_staging_headroom_warning,
)
from sglang.test.test_utils import CustomTestCase

_LOGGER = "sglang.srt.mem_cache.unified_cache.storage_attachment"


def _warning(**overrides):
    kwargs = dict(
        storage_backend="file",
        host_tokens=17_439,  # --hicache-size 2 on Qwen3-1.7B, the issue's config
        device_tokens=20_000,  # --max-total-tokens 20000
        prefetch_threshold=256,
        write_policy="write_through",
        host_memory_mode="cache",
    )
    kwargs.update(overrides)
    return storage_staging_headroom_warning(**kwargs)


class TestStorageStagingHeadroomWarning(CustomTestCase):
    def test_host_pool_not_larger_than_device_pool_warns_with_the_numbers(self):
        message = _warning()
        self.assertIsNotNone(message)
        for fragment in (
            "17439 tokens",
            "2561 tokens smaller than the device pool (20000 tokens)",
            "host_capacity",
            "--hicache-ratio",
        ):
            self.assertIn(fragment, message)

    def test_headroom_below_the_prefetch_threshold_warns(self):
        # 100 tokens of headroom: a 256-token prefetch can never be staged under pressure.
        message = _warning(host_tokens=20_100)
        self.assertIsNotNone(message)
        self.assertIn("by only 100 tokens, below the 256-token", message)

    def test_headroom_at_the_prefetch_threshold_is_silent(self):
        self.assertIsNone(_warning(host_tokens=20_256))
        # --hicache-ratio 1.1 and 2 from the reproduction both recall from storage.
        self.assertIsNone(_warning(host_tokens=22_001))
        self.assertIsNone(_warning(host_tokens=34_878))

    def test_write_back_and_buffer_only_are_not_affected(self):
        # write_back reclaims host duplicates; buffer-only mode stages outside the tree.
        self.assertIsNone(_warning(write_policy="write_back"))
        self.assertIsNone(_warning(host_memory_mode="buffer_only"))
        # write_through_selective shares write_through's eviction rules.
        self.assertIsNotNone(_warning(write_policy="write_through_selective"))


class TestShouldLogStorageHitDrop(CustomTestCase):
    def test_first_three_then_every_hundredth(self):
        logged = [n for n in range(1, 401) if should_log_storage_hit_drop(n)]
        self.assertEqual(logged, [1, 2, 3, 100, 200, 300, 400])


class TestApplyRuntimeConfigWarns(CustomTestCase):
    """The warning fires once, at the moment storage turns on, with the live pool sizes."""

    def _cache(self, *, host_tokens, device_tokens, write_policy="write_through"):
        cache = MagicMock()
        cache.enable_storage = False
        cache.page_size = 1
        cache.host_memory_mode = "cache"
        cache.tree_core.backfill_missing_hash_values.return_value = 0
        cache.cache_controller.write_policy = write_policy
        cache.cache_controller.mem_pool_host.logical_size = host_tokens
        allocator = MagicMock(spec=["size"])
        allocator.size = device_tokens
        cache.token_to_kv_pool_allocator = allocator
        return cache

    def _apply(self, cache, enable_storage=True):
        StorageAttachment(cache).apply_runtime_config(
            storage_backend="file",
            prefetch_threshold=256,
            prefetch_timeout_base=1.0,
            prefetch_timeout_per_ki_token=1.0,
            hicache_storage_pass_prefix_keys=False,
            enable_storage=enable_storage,
            enable_storage_metrics=False,
            extra_metric_labels=None,
        )

    def test_warns_when_storage_turns_on_with_a_small_host_pool(self):
        cache = self._cache(host_tokens=17_439, device_tokens=20_000)
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._apply(cache)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("storage backend 'file'", logs.output[0])
        self.assertIn("17439 tokens", logs.output[0])
        self.assertTrue(cache.enable_storage)

    def test_silent_with_a_host_pool_above_the_device_pool(self):
        cache = self._cache(host_tokens=40_000, device_tokens=20_000)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._apply(cache)

    def test_silent_when_storage_is_already_on_or_being_turned_off(self):
        cache = self._cache(host_tokens=17_439, device_tokens=20_000)
        cache.enable_storage = True
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._apply(cache)  # re-applying knobs with storage already on
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._apply(cache, enable_storage=False)  # detach


if __name__ == "__main__":
    unittest.main(verbosity=2)
