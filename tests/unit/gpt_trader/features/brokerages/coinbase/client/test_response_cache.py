"""Tests for ResponseCache HTTP caching."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpt_trader.features.brokerages.coinbase.client.response_cache import (
    CachedResponse,
    ResponseCache,
)


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_cache_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = ResponseCache()
        data = {"products": [{"id": "BTC-USD"}]}

        cache.set("/api/v3/products", data)
        result = cache.get("/api/v3/products")

        assert result == data

    def test_cache_miss_returns_none(self) -> None:
        """Test that cache miss returns None."""
        cache = ResponseCache()
        result = cache.get("/api/v3/nonexistent")

        assert result is None

    def test_cache_disabled_returns_none(self) -> None:
        """Test that disabled cache always returns None."""
        cache = ResponseCache(enabled=False)
        cache.set("/api/v3/products", {"data": "test"})

        result = cache.get("/api/v3/products")
        assert result is None

    def test_cache_expiration(self, monkeypatch) -> None:
        """Test that expired entries return None."""
        import gpt_trader.features.brokerages.coinbase.client.response_cache as cache_module

        # Control time for expiration testing
        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cache_module, "time", FakeTime)

        cache = ResponseCache(
            ENDPOINT_TTLS={},  # Empty to use default_ttl
            default_ttl=10.0,  # 10 second TTL
        )

        cache.set("/api/v3/unique_test_endpoint", {"data": "test"})

        # Should be cached immediately
        assert cache.get("/api/v3/unique_test_endpoint") is not None

        # Advance time past TTL
        current_time[0] = 1015.0  # 15 seconds later

        # Should now be expired
        assert cache.get("/api/v3/unique_test_endpoint") is None

    def test_endpoint_specific_ttls(self) -> None:
        """Test that different endpoints get different TTLs."""
        cache = ResponseCache()

        # Products should have 5 minute TTL
        assert cache._get_ttl_for_endpoint("/api/v3/products") == 300.0

        # Ticker should have 5 second TTL
        assert cache._get_ttl_for_endpoint("/api/v3/ticker") == 5.0

        # Accounts should have 60 second TTL
        assert cache._get_ttl_for_endpoint("/api/v3/accounts") == 60.0

        # Unknown endpoint uses default
        assert cache._get_ttl_for_endpoint("/api/v3/unknown") == cache.default_ttl

    def test_cache_invalidation_by_pattern(self) -> None:
        """Test invalidating entries by pattern."""
        cache = ResponseCache()
        cache.set("/api/v3/orders", {"orders": []})
        cache.set("/api/v3/orders/123", {"order_id": "123"})
        cache.set("/api/v3/products", {"products": []})

        # Invalidate orders
        count = cache.invalidate("**/orders*")
        assert count == 2

        # Orders should be gone
        assert cache.get("/api/v3/orders") is None
        assert cache.get("/api/v3/orders/123") is None

        # Products should remain
        assert cache.get("/api/v3/products") is not None

    def test_cache_invalidate_all(self) -> None:
        """Test clearing entire cache."""
        cache = ResponseCache()
        cache.set("/api/v3/orders", {"orders": []})
        cache.set("/api/v3/products", {"products": []})

        count = cache.invalidate_all()
        assert count == 2

        assert cache.get("/api/v3/orders") is None
        assert cache.get("/api/v3/products") is None

    def test_cache_max_size_eviction(self) -> None:
        """Test that oldest entries are evicted when max size is reached."""
        cache = ResponseCache(max_size=3)

        cache.set("/api/v3/item1", {"id": 1})
        time.sleep(0.01)
        cache.set("/api/v3/item2", {"id": 2})
        time.sleep(0.01)
        cache.set("/api/v3/item3", {"id": 3})
        time.sleep(0.01)

        # Adding fourth item should evict oldest (item1)
        cache.set("/api/v3/item4", {"id": 4})

        assert cache.get("/api/v3/item1") is None  # Evicted
        assert cache.get("/api/v3/item2") is not None
        assert cache.get("/api/v3/item3") is not None
        assert cache.get("/api/v3/item4") is not None

    def test_cache_stats(self) -> None:
        """Test cache statistics tracking."""
        cache = ResponseCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # Add item
        cache.set("/api/v3/test", {"data": "test"})

        # Hit
        cache.get("/api/v3/test")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["size"] == 1

        # Miss
        cache.get("/api/v3/nonexistent")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_thread_safety(self) -> None:
        """Test that cache is thread-safe."""
        cache = ResponseCache()
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(100):
                    cache.set(f"/api/v3/item{n}_{i}", {"n": n, "i": i})
            except Exception as e:
                errors.append(e)

        def reader(n: int) -> None:
            try:
                for _ in range(100):
                    cache.get(f"/api/v3/item{n}_50")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(writer, i))
                futures.append(executor.submit(reader, i))

            for f in futures:
                f.result()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_cache_key_normalization(self) -> None:
        """Test that cache keys are normalized."""
        cache = ResponseCache()
        cache.set("/api/v3/Products", {"data": "test"})

        # Should match with different case/slashes
        assert cache.get("api/v3/products") is not None
        assert cache.get("/API/V3/PRODUCTS/") is not None


class TestCachedResponse:
    """Tests for CachedResponse dataclass."""

    def test_cached_response_creation(self) -> None:
        """Test creating a CachedResponse."""
        data = {"test": "data"}
        response = CachedResponse(
            data=data,
            timestamp=time.time(),
            endpoint="/api/test",
        )

        assert response.data == data
        assert response.hits == 0

    def test_cached_response_hit_counter(self) -> None:
        """Test that hit counter increments."""
        response = CachedResponse(
            data={"test": "data"},
            timestamp=time.time(),
            endpoint="/api/test",
        )

        assert response.hits == 0
        response.hits += 1
        assert response.hits == 1
