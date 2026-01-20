"""Tests for CircuitBreakerRegistry behavior and API resilience integration."""

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry class."""

    def test_get_breaker_creates_new(self) -> None:
        """Test that get_breaker creates new breakers as needed."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get_breaker("/api/v3/orders")
        assert breaker is not None
        assert breaker.state == CircuitState.CLOSED

    def test_same_category_shares_breaker(self) -> None:
        """Test that endpoints in same category share a breaker."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_breaker("/api/v3/orders")
        breaker2 = registry.get_breaker("/api/v3/orders/123")

        assert breaker1 is breaker2

    def test_different_categories_different_breakers(self) -> None:
        """Test that different categories have different breakers."""
        registry = CircuitBreakerRegistry()

        orders_breaker = registry.get_breaker("/api/v3/orders")
        accounts_breaker = registry.get_breaker("/api/v3/accounts")

        assert orders_breaker is not accounts_breaker

    def test_can_proceed_delegates_to_breaker(self) -> None:
        """Test that can_proceed delegates to the appropriate breaker."""
        registry = CircuitBreakerRegistry(default_failure_threshold=2)

        assert registry.can_proceed("/api/v3/orders") is True

        # Trip the orders breaker
        registry.record_failure("/api/v3/orders")
        registry.record_failure("/api/v3/orders")

        assert registry.can_proceed("/api/v3/orders") is False
        assert registry.can_proceed("/api/v3/accounts") is True  # Different category

    def test_disabled_registry_always_allows(self) -> None:
        """Test that disabled registry always allows requests."""
        registry = CircuitBreakerRegistry(enabled=False, default_failure_threshold=1)

        # Even after recording failure, should allow
        registry.record_failure("/api/v3/orders")
        registry.record_failure("/api/v3/orders")

        assert registry.can_proceed("/api/v3/orders") is True

    def test_get_all_status(self) -> None:
        """Test getting status of all breakers."""
        registry = CircuitBreakerRegistry()

        registry.get_breaker("/api/v3/orders")
        registry.get_breaker("/api/v3/accounts")

        status = registry.get_all_status()
        assert "orders" in status
        assert "accounts" in status

    def test_reset_all(self) -> None:
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry(default_failure_threshold=1)

        # Trip both breakers
        registry.record_failure("/api/v3/orders")
        registry.record_failure("/api/v3/accounts")

        assert registry.can_proceed("/api/v3/orders") is False
        assert registry.can_proceed("/api/v3/accounts") is False

        registry.reset_all()

        assert registry.can_proceed("/api/v3/orders") is True
        assert registry.can_proceed("/api/v3/accounts") is True


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
