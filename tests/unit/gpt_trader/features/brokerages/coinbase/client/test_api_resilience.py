"""Integration tests for Coinbase API resilience: cache, circuit breaker, metrics, priority."""

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)
from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.client.priority import (
    PriorityManager,
    RequestPriority,
)
from gpt_trader.features.brokerages.coinbase.client.response_cache import ResponseCache

# ============================================================================
# Cache Integration Tests
# ============================================================================


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


# ============================================================================
# Full Flow Integration Tests
# ============================================================================


class TestAPIResilienceFlows:
    """Integration tests for complete request flows."""

    def test_full_request_flow_simulation(self, monkeypatch) -> None:
        """Test a complete request flow through all resilience components."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module
        import gpt_trader.features.brokerages.coinbase.client.response_cache as cache_module

        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)
        monkeypatch.setattr(cache_module, "time", FakeTime)

        cache = ResponseCache(enabled=True)
        circuit_breaker = CircuitBreakerRegistry(
            default_failure_threshold=3,
            enabled=True,
        )
        metrics = APIMetricsCollector(enabled=True)
        priority = PriorityManager(
            threshold_high=0.7,
            enabled=True,
        )

        path = "/api/v3/accounts"
        rate_limit_usage = 0.5

        assert priority.should_allow(path, rate_limit_usage) is True
        assert circuit_breaker.can_proceed(path) is True

        cached = cache.get(path)
        assert cached is None

        response_data = {"accounts": [{"id": "123", "balance": "1000.00"}]}
        latency = 75.0
        metrics.record_request(path, latency)
        cache.set(path, response_data)
        circuit_breaker.record_success(path)

        assert cache.get(path) == response_data
        assert metrics.get_summary()["total_requests"] == 1
        assert circuit_breaker.get_breaker(path).state == CircuitState.CLOSED

        cached = cache.get(path)
        assert cached == response_data

        cache_stats = cache.get_stats()
        assert cache_stats["hits"] == 2
        assert cache_stats["misses"] == 1

    def test_disabled_components_allow_all(self) -> None:
        """Test that disabled components don't interfere with requests."""
        cache = ResponseCache(enabled=False)
        circuit_breaker = CircuitBreakerRegistry(enabled=False)
        metrics = APIMetricsCollector(enabled=False)
        priority = PriorityManager(enabled=False)

        path = "/api/v3/products"

        cache.set(path, {"data": "test"})
        assert cache.get(path) is None

        for _ in range(10):
            circuit_breaker.record_failure(path)
        assert circuit_breaker.can_proceed(path) is True

        metrics.record_request(path, 100.0)
        assert metrics.get_summary()["total_requests"] == 0

        assert priority.should_allow(path, 1.0) is True

    def test_error_handling_updates_metrics_and_circuit(self) -> None:
        """Test that errors properly update both metrics and circuit breaker."""
        metrics = APIMetricsCollector(enabled=True)
        circuit_breaker = CircuitBreakerRegistry(
            default_failure_threshold=2,
            enabled=True,
        )

        path = "/api/v3/orders"

        metrics.record_request(path, 100.0, error=True)
        circuit_breaker.record_failure(path)

        assert metrics.get_summary()["total_errors"] == 1
        assert circuit_breaker.can_proceed(path) is True

        metrics.record_request(path, 100.0, error=True)
        circuit_breaker.record_failure(path)

        assert metrics.get_summary()["total_errors"] == 2
        assert circuit_breaker.can_proceed(path) is False


# ============================================================================
# Priority and Metrics Tests
# ============================================================================


class TestAPIResilienceMetricsAndPriority:
    """Tests for metrics and priority manager integration."""

    def test_priority_blocks_low_under_pressure(self) -> None:
        """Test that low priority requests are blocked under rate limit pressure."""
        priority_manager = PriorityManager(
            threshold_high=0.7,
            threshold_critical=0.85,
            enabled=True,
        )

        products_path = "/api/v3/products"
        orders_path = "/api/v3/orders"

        assert priority_manager.get_priority(products_path) == RequestPriority.LOW
        assert priority_manager.get_priority(orders_path) == RequestPriority.CRITICAL

        assert priority_manager.should_allow(products_path, 0.80) is False
        assert priority_manager.should_allow(orders_path, 0.80) is True

    def test_metrics_tracks_across_endpoints(self) -> None:
        """Test that metrics correctly aggregate across different endpoints."""
        metrics = APIMetricsCollector()

        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders/123", 150.0, error=True)
        metrics.record_request("/api/v3/accounts", 200.0)
        metrics.record_request("/api/v3/products", 50.0)

        summary = metrics.get_summary()

        assert summary["total_requests"] == 4
        assert summary["total_errors"] == 1
        assert summary["avg_latency_ms"] == 125.0

        assert summary["endpoints"]["orders"]["total_calls"] == 2
        assert summary["endpoints"]["orders"]["total_errors"] == 1
        assert summary["endpoints"]["accounts"]["total_calls"] == 1
        assert summary["endpoints"]["products"]["total_calls"] == 1

    def test_rate_limit_tracking(self) -> None:
        """Test that rate limit hits are tracked in metrics."""
        metrics = APIMetricsCollector(enabled=True)

        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders", 100.0)

        metrics.record_request("/api/v3/orders", 50.0, error=True, rate_limited=True)

        summary = metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["rate_limit_hits"] == 1
