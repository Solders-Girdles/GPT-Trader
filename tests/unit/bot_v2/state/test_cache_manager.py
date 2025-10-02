"""
Unit tests for StateCacheManager

Tests cache management, metadata tracking, and eviction logic in isolation.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from bot_v2.state.cache_manager import StateCacheManager
from bot_v2.state.state_manager import StateCategory, StateConfig


@pytest.fixture
def config():
    """Create test config."""
    return StateConfig(cache_size_mb=0.001)  # 1KB for testing


@pytest.fixture
def cache_manager(config):
    """Create cache manager instance."""
    return StateCacheManager(config=config)


class TestCacheOperations:
    """Test basic cache operations."""

    def test_get_returns_none_for_missing_key(self, cache_manager):
        """Should return None for non-existent key."""
        assert cache_manager.get("missing_key") is None

    def test_set_and_get_value(self, cache_manager):
        """Should store and retrieve value."""
        cache_manager.set("test_key", {"value": 123})
        assert cache_manager.get("test_key") == {"value": 123}

    def test_get_updates_access_history(self, cache_manager):
        """Getting a value should update access history."""
        cache_manager.set("test_key", "value")
        initial_history_length = len(cache_manager.get_access_history("test_key"))

        cache_manager.get("test_key")

        assert len(cache_manager.get_access_history("test_key")) > initial_history_length

    def test_contains_returns_true_for_existing_key(self, cache_manager):
        """Should return True for cached key."""
        cache_manager.set("test_key", "value")
        assert cache_manager.contains("test_key")

    def test_contains_returns_false_for_missing_key(self, cache_manager):
        """Should return False for non-existent key."""
        assert not cache_manager.contains("missing_key")

    def test_delete_removes_key(self, cache_manager):
        """Should remove key from cache."""
        cache_manager.set("test_key", "value")
        cache_manager.delete("test_key")
        assert not cache_manager.contains("test_key")

    def test_delete_removes_metadata(self, cache_manager):
        """Should remove metadata when deleting key."""
        cache_manager.set("test_key", "value")
        cache_manager.update_metadata(
            "test_key", StateCategory.HOT, 100, "checksum123"
        )

        cache_manager.delete("test_key")

        assert cache_manager.get_metadata("test_key") is None

    def test_delete_removes_access_history(self, cache_manager):
        """Should remove access history when deleting key."""
        cache_manager.set("test_key", "value")
        cache_manager.update_access_history("test_key")

        cache_manager.delete("test_key")

        assert cache_manager.get_access_history("test_key") == []


class TestMetadataManagement:
    """Test metadata tracking."""

    def test_update_metadata_stores_all_fields(self, cache_manager):
        """Should store all metadata fields."""
        cache_manager.update_metadata(
            key="test_key",
            category=StateCategory.WARM,
            size_bytes=1024,
            checksum="abc123",
            ttl_seconds=3600,
        )

        metadata = cache_manager.get_metadata("test_key")
        assert metadata is not None
        assert metadata.key == "test_key"
        assert metadata.category == StateCategory.WARM
        assert metadata.size_bytes == 1024
        assert metadata.checksum == "abc123"
        assert metadata.ttl_seconds == 3600

    def test_update_metadata_without_ttl(self, cache_manager):
        """Should handle metadata without TTL."""
        cache_manager.update_metadata(
            key="test_key",
            category=StateCategory.COLD,
            size_bytes=512,
            checksum="def456",
        )

        metadata = cache_manager.get_metadata("test_key")
        assert metadata.ttl_seconds is None

    def test_get_metadata_returns_none_for_missing_key(self, cache_manager):
        """Should return None for non-existent metadata."""
        assert cache_manager.get_metadata("missing_key") is None

    def test_metadata_includes_timestamps(self, cache_manager):
        """Metadata should include creation and access timestamps."""
        before = datetime.utcnow()
        cache_manager.update_metadata(
            "test_key", StateCategory.HOT, 100, "checksum"
        )
        after = datetime.utcnow()

        metadata = cache_manager.get_metadata("test_key")
        assert before <= metadata.created_at <= after
        assert before <= metadata.last_accessed <= after


class TestAccessHistory:
    """Test access history tracking."""

    def test_update_access_history_creates_entry(self, cache_manager):
        """Should create history entry for new key."""
        cache_manager.update_access_history("test_key")
        history = cache_manager.get_access_history("test_key")
        assert len(history) == 1

    def test_update_access_history_appends_timestamps(self, cache_manager):
        """Should append timestamps to history."""
        cache_manager.update_access_history("test_key")
        cache_manager.update_access_history("test_key")
        cache_manager.update_access_history("test_key")

        history = cache_manager.get_access_history("test_key")
        assert len(history) == 3

    def test_access_history_limited_to_100_entries(self, cache_manager):
        """Should limit history to last 100 accesses."""
        for _ in range(150):
            cache_manager.update_access_history("test_key")

        history = cache_manager.get_access_history("test_key")
        assert len(history) == 100

    def test_access_history_keeps_most_recent(self, cache_manager):
        """Should keep most recent entries when truncating."""
        # Add 150 entries with timestamps
        for i in range(150):
            cache_manager.update_access_history("test_key")

        history = cache_manager.get_access_history("test_key")
        # Should have last 100 entries (most recent)
        assert len(history) == 100
        # Entries should be in chronological order
        for i in range(len(history) - 1):
            assert history[i] <= history[i + 1]

    def test_get_access_history_returns_empty_for_missing_key(self, cache_manager):
        """Should return empty list for non-existent key."""
        assert cache_manager.get_access_history("missing_key") == []


class TestCacheEviction:
    """Test cache size management and eviction."""

    def test_manage_cache_size_evicts_when_over_limit(self, cache_manager):
        """Should evict items when cache exceeds size limit."""
        # Add items that exceed 1KB limit
        for i in range(50):
            cache_manager._local_cache[f"key_{i}"] = {"data": "x" * 100}
            cache_manager.update_access_history(f"key_{i}")

        cache_manager.manage_cache_size()

        # Cache should be smaller than 50 items
        assert len(cache_manager._local_cache) < 50

    def test_evicts_least_recently_accessed_items(self):
        """Should evict LRU items first."""
        # Use larger cache for this test
        config = StateConfig(cache_size_mb=0.01)  # 10KB
        cache_manager = StateCacheManager(config=config)

        # Add items with explicitly different access times
        import time

        cache_manager._local_cache["oldest"] = {"data": "x" * 50}
        cache_manager._access_history["oldest"] = [datetime(2020, 1, 1)]

        cache_manager._local_cache["middle"] = {"data": "x" * 50}
        cache_manager._access_history["middle"] = [datetime(2023, 1, 1)]

        cache_manager._local_cache["newest"] = {"data": "x" * 50}
        cache_manager._access_history["newest"] = [datetime.utcnow()]

        # Add many items to trigger eviction
        for i in range(100):
            cache_manager._local_cache[f"key_{i}"] = {"data": "x" * 100}
            cache_manager._access_history[f"key_{i}"] = [datetime.utcnow()]

        cache_manager.manage_cache_size()

        # Oldest should definitely be evicted
        assert "oldest" not in cache_manager._local_cache

    def test_respects_config_cache_size(self):
        """Should respect configured cache size limit."""
        config = StateConfig(cache_size_mb=0.01)  # 10KB
        cache_manager = StateCacheManager(config=config)

        # Add items
        for i in range(100):
            cache_manager._local_cache[f"key_{i}"] = {"data": "x" * 1000}

        cache_manager.manage_cache_size()

        # Calculate actual cache size
        import json

        actual_size = sum(
            len(json.dumps(v, default=str).encode())
            for v in cache_manager._local_cache.values()
        )

        # Should be under or at the limit (10KB = 10240 bytes)
        assert actual_size <= 10240

    def test_set_triggers_cache_management(self, cache_manager):
        """Setting a value should trigger cache size management."""
        # Fill cache beyond limit
        for i in range(50):
            cache_manager._local_cache[f"key_{i}"] = {"data": "x" * 100}
            cache_manager._access_history[f"key_{i}"] = [datetime.utcnow()]

        # Setting new value should trigger eviction
        cache_manager.set("new_key", {"data": "new"})

        # Cache should have been managed
        assert len(cache_manager._local_cache) < 51


class TestChecksumCalculation:
    """Test checksum calculation."""

    def test_calculate_checksum_returns_sha256(self, cache_manager):
        """Should return SHA256 checksum."""
        checksum = cache_manager.calculate_checksum("test_data")
        # SHA256 produces 64 character hex string
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_same_data_produces_same_checksum(self, cache_manager):
        """Same data should produce same checksum."""
        checksum1 = cache_manager.calculate_checksum("test_data")
        checksum2 = cache_manager.calculate_checksum("test_data")
        assert checksum1 == checksum2

    def test_different_data_produces_different_checksum(self, cache_manager):
        """Different data should produce different checksums."""
        checksum1 = cache_manager.calculate_checksum("data1")
        checksum2 = cache_manager.calculate_checksum("data2")
        assert checksum1 != checksum2

    def test_handles_empty_string(self, cache_manager):
        """Should handle empty string."""
        checksum = cache_manager.calculate_checksum("")
        assert len(checksum) == 64


class TestCacheStats:
    """Test cache statistics."""

    def test_get_cache_stats_returns_empty_for_empty_cache(self, cache_manager):
        """Should return zero stats for empty cache."""
        stats = cache_manager.get_cache_stats()
        assert stats["cache_keys"] == 0
        assert stats["cache_size_bytes"] == 0

    def test_get_cache_stats_returns_accurate_counts(self, cache_manager):
        """Should return accurate key count."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")

        stats = cache_manager.get_cache_stats()
        assert stats["cache_keys"] == 3

    def test_get_cache_stats_calculates_size(self, cache_manager):
        """Should calculate total cache size in bytes."""
        cache_manager.set("key1", {"data": "x" * 100})
        cache_manager.set("key2", {"data": "y" * 50})

        stats = cache_manager.get_cache_stats()
        assert stats["cache_size_bytes"] > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handles_none_values_in_cache(self, cache_manager):
        """Should handle None as a cached value."""
        cache_manager.set("null_key", None)
        assert cache_manager.get("null_key") is None
        assert cache_manager.contains("null_key")

    def test_handles_complex_nested_objects(self, cache_manager):
        """Should handle complex nested data structures."""
        complex_data = {
            "nested": {"deeply": {"nested": {"value": [1, 2, 3]}}},
            "list": [{"a": 1}, {"b": 2}],
        }
        cache_manager.set("complex_key", complex_data)
        assert cache_manager.get("complex_key") == complex_data

    def test_delete_nonexistent_key_is_safe(self, cache_manager):
        """Deleting non-existent key should not raise error."""
        cache_manager.delete("nonexistent_key")  # Should not raise

    def test_config_with_zero_cache_size(self):
        """Should handle zero cache size configuration."""
        config = StateConfig(cache_size_mb=0)
        cache_manager = StateCacheManager(config=config)

        cache_manager.set("key", "value")
        cache_manager.manage_cache_size()

        # With 0 size, everything should be evicted
        assert len(cache_manager._local_cache) == 0
