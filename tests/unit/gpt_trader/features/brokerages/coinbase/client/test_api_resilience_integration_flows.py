"""Integration-style tests for Coinbase client resilience components together."""

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)
from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.client.priority import PriorityManager
from gpt_trader.features.brokerages.coinbase.client.response_cache import ResponseCache


class TestAPIResilienceIntegrationFlows:
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
