"""Tests for Coinbase API response caching behavior."""

from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.client.response_cache import ResponseCache


class TestAPIResilienceCache:
    """Integration tests for caching behavior."""

    def test_cache_prevents_duplicate_calls(self) -> None:
        """Test that cache prevents unnecessary API calls."""
        cache = ResponseCache()
        metrics = APIMetricsCollector()

        path = "/api/v3/products"
        cached = cache.get(path)
        assert cached is None

        metrics.record_request(path, 100.0)
        response_data = {"products": [{"id": "BTC-USD"}]}
        cache.set(path, response_data)

        cached = cache.get(path)
        assert cached == response_data

        summary = metrics.get_summary()
        assert summary["total_requests"] == 1

    def test_cache_invalidation_after_mutation(self) -> None:
        """Test that cache is invalidated after mutation operations."""
        cache = ResponseCache(enabled=True)

        cache.set("/api/v3/orders", {"orders": [{"id": "1"}]})
        cache.set("/api/v3/orders/1", {"order_id": "1", "status": "pending"})
        cache.set("/api/v3/accounts", {"accounts": []})

        assert cache.get("/api/v3/orders") is not None
        assert cache.get("/api/v3/orders/1") is not None
        assert cache.get("/api/v3/accounts") is not None

        invalidated = cache.invalidate("**/orders*")
        assert invalidated == 2

        assert cache.get("/api/v3/orders") is None
        assert cache.get("/api/v3/orders/1") is None
        assert cache.get("/api/v3/accounts") is not None
