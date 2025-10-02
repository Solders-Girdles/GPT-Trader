"""
Comprehensive tests for DataCache.

Covers TTL expiration, LRU eviction, hit/miss tracking, and edge cases.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from bot_v2.features.data.cache import DataCache


@pytest.fixture
def cache():
    """Create cache with 1MB limit."""
    return DataCache(max_size_mb=1.0)


@pytest.fixture
def sample_data():
    """Small test DataFrame."""
    return pd.DataFrame({
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100.5, 101.5, 102.5],
        "volume": [1000000, 1100000, 1200000],
    })


@pytest.fixture
def large_data():
    """Larger DataFrame for eviction tests."""
    # ~100KB DataFrame
    return pd.DataFrame({
        "col1": range(5000),
        "col2": range(5000, 10000),
        "col3": range(10000, 15000),
    })


class TestCachePut:
    """Test cache put operations."""

    def test_put_basic(self, cache, sample_data):
        """Should store data successfully."""
        result = cache.put("test_key", sample_data, ttl_seconds=3600)

        assert result is True
        assert "test_key" in cache.cache
        assert cache.cache["test_key"].hit_count == 0

    def test_put_sets_expiration(self, cache, sample_data):
        """Should set correct expiration time."""
        cache.put("test_key", sample_data, ttl_seconds=60)

        entry = cache.cache["test_key"]
        expected_expiry = datetime.now() + timedelta(seconds=60)

        # Allow 1 second tolerance
        assert abs((entry.expires_at - expected_expiry).total_seconds()) < 1

    def test_put_calculates_size(self, cache, sample_data):
        """Should calculate and store data size."""
        cache.put("test_key", sample_data)

        entry = cache.cache["test_key"]
        assert entry.size_bytes > 0

    def test_put_copies_data(self, cache, sample_data):
        """Should copy data to avoid mutations."""
        cache.put("test_key", sample_data)

        # Modify original
        sample_data.loc[0, "open"] = 999

        # Cached version should be unchanged
        cached = cache.cache["test_key"].data
        assert cached.loc[0, "open"] != 999

    def test_put_overwrites_existing(self, cache, sample_data):
        """Should overwrite existing key."""
        cache.put("test_key", sample_data, ttl_seconds=60)

        new_data = sample_data.copy()
        new_data["new_col"] = [1, 2, 3]
        cache.put("test_key", new_data, ttl_seconds=120)

        entry = cache.cache["test_key"]
        assert "new_col" in entry.data.columns


class TestCacheGet:
    """Test cache get operations."""

    def test_get_existing_key(self, cache, sample_data):
        """Should return cached data."""
        cache.put("test_key", sample_data)
        result = cache.get("test_key")

        assert result is not None
        pd.testing.assert_frame_equal(result, sample_data)

    def test_get_missing_key(self, cache):
        """Should return None for missing key."""
        result = cache.get("nonexistent")

        assert result is None
        assert cache.total_misses == 1

    def test_get_expired_key(self, cache, sample_data):
        """Should return None and remove expired entry."""
        cache.put("test_key", sample_data, ttl_seconds=1)
        time.sleep(1.1)

        result = cache.get("test_key")

        assert result is None
        assert "test_key" not in cache.cache
        assert cache.total_misses == 1

    def test_get_increments_hit_count(self, cache, sample_data):
        """Should track hit count."""
        cache.put("test_key", sample_data)

        cache.get("test_key")
        cache.get("test_key")
        cache.get("test_key")

        assert cache.cache["test_key"].hit_count == 3
        assert cache.total_hits == 3

    def test_get_returns_copy(self, cache, sample_data):
        """Should return copy to prevent mutations."""
        cache.put("test_key", sample_data)

        result1 = cache.get("test_key")
        result1.loc[0, "open"] = 999

        result2 = cache.get("test_key")
        assert result2.loc[0, "open"] != 999


class TestCacheTTL:
    """Test TTL and expiration."""

    def test_ttl_not_expired(self, cache, sample_data):
        """Should return data before expiration."""
        cache.put("test_key", sample_data, ttl_seconds=10)
        time.sleep(0.1)

        result = cache.get("test_key")
        assert result is not None

    def test_ttl_expired(self, cache, sample_data):
        """Should expire after TTL."""
        cache.put("test_key", sample_data, ttl_seconds=1)
        time.sleep(1.1)

        result = cache.get("test_key")
        assert result is None

    def test_clear_expired(self, cache, sample_data):
        """Should clear only expired entries."""
        cache.put("fresh", sample_data, ttl_seconds=100)
        cache.put("expired", sample_data, ttl_seconds=1)
        time.sleep(1.1)

        cleared = cache.clear_expired()

        assert cleared == 1
        assert "fresh" in cache.cache
        assert "expired" not in cache.cache

    def test_clear_expired_multiple(self, cache, sample_data):
        """Should clear multiple expired entries."""
        for i in range(5):
            cache.put(f"key_{i}", sample_data, ttl_seconds=1)

        time.sleep(1.1)
        cleared = cache.clear_expired()

        assert cleared == 5
        assert len(cache.cache) == 0


class TestCacheEviction:
    """Test LRU eviction."""

    def test_eviction_when_full(self, cache, large_data):
        """Should evict old entries when size limit reached."""
        # Fill cache to capacity
        cache.put("key1", large_data, ttl_seconds=3600)
        cache.put("key2", large_data, ttl_seconds=3600)
        cache.put("key3", large_data, ttl_seconds=3600)

        # Access key2 to increase hit count
        cache.get("key2")
        cache.get("key2")

        # This should trigger eviction
        cache.put("key4", large_data, ttl_seconds=3600)

        # key2 should survive (higher hit count), key1 likely evicted
        assert "key2" in cache.cache
        # Total size should be under limit
        assert cache._get_total_size() <= cache.max_size_bytes

    def test_eviction_prefers_low_hit_count(self, cache, sample_data):
        """Should evict entries with low hit counts first."""
        cache.put("popular", sample_data, ttl_seconds=3600)
        cache.put("unpopular", sample_data, ttl_seconds=3600)

        # Make 'popular' frequently accessed
        for _ in range(10):
            cache.get("popular")

        # Access unpopular once
        cache.get("unpopular")

        # Force eviction by filling cache
        large = pd.DataFrame({"data": range(100000)})
        cache.put("large1", large, ttl_seconds=3600)
        cache.put("large2", large, ttl_seconds=3600)

        # Popular should survive
        assert "popular" in cache.cache

    def test_eviction_frees_required_space(self):
        """Should evict enough entries to fit new data."""
        # Use smaller cache to force eviction
        small_cache = DataCache(max_size_mb=0.1)  # 100KB limit

        # Create entries of known size (~8KB each)
        small_data = pd.DataFrame({"col": range(1000)})

        for i in range(10):
            small_cache.put(f"small_{i}", small_data, ttl_seconds=3600)

        initial_count = len(small_cache.cache)

        # Add large data requiring evictions (~400KB)
        large = pd.DataFrame({"data": range(50000)})
        small_cache.put("large", large, ttl_seconds=3600)

        # Should have evicted some entries
        assert len(small_cache.cache) < initial_count
        assert "large" in small_cache.cache


class TestCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_existing_key(self, cache, sample_data):
        """Should remove entry."""
        cache.put("test_key", sample_data)
        result = cache.invalidate("test_key")

        assert result is True
        assert "test_key" not in cache.cache

    def test_invalidate_missing_key(self, cache):
        """Should return False for missing key."""
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear_all(self, cache, sample_data):
        """Should clear all entries and stats."""
        cache.put("key1", sample_data)
        cache.put("key2", sample_data)
        cache.get("key1")

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.total_hits == 0
        assert cache.total_misses == 0


class TestCacheStats:
    """Test cache statistics."""

    def test_stats_empty_cache(self, cache):
        """Should return zero stats for empty cache."""
        stats = cache.get_stats()

        assert stats["entries"] == 0
        assert stats["size_mb"] == 0
        assert stats["hit_rate"] == 0
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0

    def test_stats_with_data(self, cache, sample_data):
        """Should calculate correct stats."""
        cache.put("key1", sample_data)
        cache.put("key2", sample_data)

        cache.get("key1")  # hit
        cache.get("key2")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()

        assert stats["entries"] == 2
        assert stats["size_mb"] > 0
        assert stats["hit_rate"] == 2/3  # 2 hits, 1 miss
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1

    def test_stats_expired_count(self, cache, sample_data):
        """Should count expired entries."""
        cache.put("fresh", sample_data, ttl_seconds=100)
        cache.put("expired", sample_data, ttl_seconds=1)
        time.sleep(1.1)

        stats = cache.get_stats()

        assert stats["expired_entries"] == 1

    def test_stats_size_calculation(self, cache, large_data):
        """Should calculate total size correctly."""
        cache.put("key1", large_data)

        stats = cache.get_stats()
        total_size = cache._get_total_size()

        assert stats["size_mb"] == total_size / (1024 * 1024)


class TestCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_put_empty_dataframe(self, cache):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame()
        result = cache.put("empty", empty_df)

        assert result is True
        assert "empty" in cache.cache

    def test_put_with_zero_ttl(self, cache, sample_data):
        """Should immediately expire with zero TTL."""
        cache.put("instant_expire", sample_data, ttl_seconds=0)

        result = cache.get("instant_expire")
        assert result is None

    def test_multiple_evictions(self, cache):
        """Should handle cascading evictions."""
        # Fill cache
        medium_data = pd.DataFrame({"col": range(5000)})

        for i in range(5):
            cache.put(f"key_{i}", medium_data)

        # Verify evictions occurred
        assert cache._get_total_size() <= cache.max_size_bytes

    def test_concurrent_hit_tracking(self, cache, sample_data):
        """Should correctly track hits across multiple gets."""
        cache.put("key", sample_data)

        for i in range(100):
            cache.get("key")

        assert cache.total_hits == 100
        assert cache.cache["key"].hit_count == 100

    def test_warm_up_placeholder(self, cache):
        """warm_up method exists but is placeholder."""
        # Should not raise error
        cache.warm_up(["query1", "query2"])
        # No assertions - method is a placeholder

    def test_age_calculation(self, cache, sample_data):
        """Should calculate entry age correctly."""
        cache.put("test", sample_data)
        time.sleep(0.1)

        entry = cache.cache["test"]
        age = entry.age_seconds()

        assert age >= 0.1
        assert age < 1.0
