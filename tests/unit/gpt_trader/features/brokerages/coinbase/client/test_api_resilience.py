"""Integration tests for API Resilience components.

Tests that cache, circuit breaker, metrics, and priority work together
correctly when integrated into the Coinbase client.
"""

import pytest

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)
from gpt_trader.features.brokerages.coinbase.client.metrics import (
    APIMetricsCollector,
)
from gpt_trader.features.brokerages.coinbase.client.priority import (
    PriorityManager,
    RequestPriority,
)
from gpt_trader.features.brokerages.coinbase.client.response_cache import (
    ResponseCache,
)


class TestAPIResilienceIntegration:
    """Integration tests for all resilience components working together."""

    def test_cache_prevents_duplicate_calls(self) -> None:
        """Test that cache prevents unnecessary API calls."""
        cache = ResponseCache()
        metrics = APIMetricsCollector()

        # Simulate first call
        path = "/api/v3/products"
        cached = cache.get(path)
        assert cached is None

        # Record the API call and cache response
        metrics.record_request(path, 100.0)
        response_data = {"products": [{"id": "BTC-USD"}]}
        cache.set(path, response_data)

        # Second request should hit cache, no API call
        cached = cache.get(path)
        assert cached == response_data

        # Only one API call should have been recorded
        summary = metrics.get_summary()
        assert summary["total_requests"] == 1

    def test_circuit_breaker_trips_after_failures(self, monkeypatch) -> None:
        """Test that circuit breaker opens after threshold failures."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module

        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)

        registry = CircuitBreakerRegistry(
            default_failure_threshold=3,
            enabled=True,
        )

        path = "/api/v3/orders"

        # First 2 failures should still allow
        registry.record_failure(path)
        assert registry.can_proceed(path) is True

        registry.record_failure(path)
        assert registry.can_proceed(path) is True

        # Third failure should trip the circuit
        registry.record_failure(path)
        assert registry.can_proceed(path) is False

    def test_circuit_breaker_recovers_in_half_open(self, monkeypatch) -> None:
        """Test that circuit breaker transitions through half-open to closed."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module

        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)

        registry = CircuitBreakerRegistry(
            default_failure_threshold=2,
            default_recovery_timeout=10.0,
            default_success_threshold=2,
            enabled=True,
        )

        path = "/api/v3/orders"

        # Trip the circuit
        registry.record_failure(path)
        registry.record_failure(path)
        assert registry.can_proceed(path) is False

        # Advance time past recovery
        current_time[0] = 1015.0

        # Should now be in half-open, allowing requests
        breaker = registry.get_breaker(path)
        assert breaker.state == CircuitState.HALF_OPEN
        assert registry.can_proceed(path) is True

        # Two successes should close the circuit
        registry.record_success(path)
        registry.record_success(path)
        assert breaker.state == CircuitState.CLOSED

    def test_priority_blocks_low_under_pressure(self) -> None:
        """Test that low priority requests are blocked under rate limit pressure."""
        priority_manager = PriorityManager(
            threshold_high=0.7,
            threshold_critical=0.85,
            enabled=True,
        )

        # At 80% usage, LOW should be blocked
        products_path = "/api/v3/products"
        orders_path = "/api/v3/orders"

        assert priority_manager.get_priority(products_path) == RequestPriority.LOW
        assert priority_manager.get_priority(orders_path) == RequestPriority.CRITICAL

        # LOW blocked, CRITICAL allowed at high usage
        assert priority_manager.should_allow(products_path, 0.80) is False
        assert priority_manager.should_allow(orders_path, 0.80) is True

    def test_metrics_tracks_across_endpoints(self) -> None:
        """Test that metrics correctly aggregate across different endpoints."""
        metrics = APIMetricsCollector()

        # Record requests to different endpoints
        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders/123", 150.0, error=True)
        metrics.record_request("/api/v3/accounts", 200.0)
        metrics.record_request("/api/v3/products", 50.0)

        summary = metrics.get_summary()

        # Overall stats
        assert summary["total_requests"] == 4
        assert summary["total_errors"] == 1
        assert summary["avg_latency_ms"] == 125.0  # (100+150+200+50)/4

        # Endpoint-specific stats
        assert summary["endpoints"]["orders"]["total_calls"] == 2
        assert summary["endpoints"]["orders"]["total_errors"] == 1
        assert summary["endpoints"]["accounts"]["total_calls"] == 1
        assert summary["endpoints"]["products"]["total_calls"] == 1

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

        # Initialize all components
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

        # Simulate a request flow
        path = "/api/v3/accounts"
        rate_limit_usage = 0.5  # 50% usage

        # Step 1: Check priority
        assert priority.should_allow(path, rate_limit_usage) is True

        # Step 2: Check circuit breaker
        assert circuit_breaker.can_proceed(path) is True

        # Step 3: Check cache (miss on first request)
        cached = cache.get(path)
        assert cached is None

        # Step 4: Simulate successful API call
        response_data = {"accounts": [{"id": "123", "balance": "1000.00"}]}
        latency = 75.0
        metrics.record_request(path, latency)
        cache.set(path, response_data)
        circuit_breaker.record_success(path)

        # Step 5: Verify state after successful request
        assert cache.get(path) == response_data
        assert metrics.get_summary()["total_requests"] == 1
        assert circuit_breaker.get_breaker(path).state == CircuitState.CLOSED

        # Simulate follow-up request (should hit cache)
        cached = cache.get(path)
        assert cached == response_data

        # Cache stats should show hits:
        # - First get() after set() = 1 hit (in step 5)
        # - Second get() = 2 hits (follow-up request)
        cache_stats = cache.get_stats()
        assert cache_stats["hits"] == 2
        assert cache_stats["misses"] == 1  # First request (step 3) was a miss

    def test_cache_invalidation_after_mutation(self) -> None:
        """Test that cache is invalidated after mutation operations."""
        cache = ResponseCache(enabled=True)

        # Cache some data
        cache.set("/api/v3/orders", {"orders": [{"id": "1"}]})
        cache.set("/api/v3/orders/1", {"order_id": "1", "status": "pending"})
        cache.set("/api/v3/accounts", {"accounts": []})

        # Verify data is cached
        assert cache.get("/api/v3/orders") is not None
        assert cache.get("/api/v3/orders/1") is not None
        assert cache.get("/api/v3/accounts") is not None

        # Simulate a mutation (order placement) - invalidate orders cache
        invalidated = cache.invalidate("**/orders*")
        assert invalidated == 2  # Both order entries

        # Orders should be gone, accounts should remain
        assert cache.get("/api/v3/orders") is None
        assert cache.get("/api/v3/orders/1") is None
        assert cache.get("/api/v3/accounts") is not None

    def test_disabled_components_allow_all(self) -> None:
        """Test that disabled components don't interfere with requests."""
        cache = ResponseCache(enabled=False)
        circuit_breaker = CircuitBreakerRegistry(enabled=False)
        metrics = APIMetricsCollector(enabled=False)
        priority = PriorityManager(enabled=False)

        path = "/api/v3/products"

        # Disabled cache always returns None (miss)
        cache.set(path, {"data": "test"})
        assert cache.get(path) is None

        # Disabled circuit breaker always allows
        for _ in range(10):
            circuit_breaker.record_failure(path)
        assert circuit_breaker.can_proceed(path) is True

        # Disabled metrics doesn't record
        metrics.record_request(path, 100.0)
        assert metrics.get_summary()["total_requests"] == 0

        # Disabled priority always allows
        assert priority.should_allow(path, 1.0) is True  # Even at 100% usage

    def test_error_handling_updates_metrics_and_circuit(self) -> None:
        """Test that errors properly update both metrics and circuit breaker."""
        metrics = APIMetricsCollector(enabled=True)
        circuit_breaker = CircuitBreakerRegistry(
            default_failure_threshold=2,
            enabled=True,
        )

        path = "/api/v3/orders"

        # First error
        metrics.record_request(path, 100.0, error=True)
        circuit_breaker.record_failure(path)

        assert metrics.get_summary()["total_errors"] == 1
        assert circuit_breaker.can_proceed(path) is True  # Still closed

        # Second error trips circuit
        metrics.record_request(path, 100.0, error=True)
        circuit_breaker.record_failure(path)

        assert metrics.get_summary()["total_errors"] == 2
        assert circuit_breaker.can_proceed(path) is False  # Now open

    def test_rate_limit_tracking(self) -> None:
        """Test that rate limit hits are tracked in metrics."""
        metrics = APIMetricsCollector(enabled=True)

        # Simulate some normal requests
        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders", 100.0)

        # Simulate a rate limit hit
        metrics.record_request("/api/v3/orders", 50.0, error=True, rate_limited=True)

        summary = metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["rate_limit_hits"] == 1
