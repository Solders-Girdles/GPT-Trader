"""Tests for Coinbase API circuit breaker behavior."""

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)


class TestAPIResilienceCircuitBreaker:
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

        registry.record_failure(path)
        assert registry.can_proceed(path) is True

        registry.record_failure(path)
        assert registry.can_proceed(path) is True

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

        registry.record_failure(path)
        registry.record_failure(path)
        assert registry.can_proceed(path) is False

        current_time[0] = 1015.0

        breaker = registry.get_breaker(path)
        assert breaker.state == CircuitState.HALF_OPEN
        assert registry.can_proceed(path) is True

        registry.record_success(path)
        registry.record_success(path)
        assert breaker.state == CircuitState.CLOSED
