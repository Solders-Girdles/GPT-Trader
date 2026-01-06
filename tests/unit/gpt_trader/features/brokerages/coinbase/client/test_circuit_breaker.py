"""Tests for CircuitBreaker pattern implementation."""

from concurrent.futures import ThreadPoolExecutor

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit breaker starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_proceed() is True

    def test_failures_trip_breaker(self) -> None:
        """Test that threshold failures trip the breaker to open."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.can_proceed() is False

    def test_success_resets_failure_count(self) -> None:
        """Test that success resets the failure counter."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        # Counter should be reset
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

    def test_recovery_to_half_open(self, monkeypatch) -> None:
        """Test that breaker transitions to half-open after recovery timeout."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module

        # Start at a known time
        current_time = [1000.0]  # Use list for mutability in closure

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Advance time past recovery timeout
        current_time[0] = 1015.0

        # Should transition to half-open on state check
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_proceed() is True

    def test_half_open_to_closed_on_success(self, monkeypatch) -> None:
        """Test that half-open transitions to closed after success threshold."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module

        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)

        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=10.0,
            success_threshold=2,
        )

        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Advance time past recovery
        current_time[0] = 1015.0
        assert breaker.state == CircuitState.HALF_OPEN

        # First success
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self, monkeypatch) -> None:
        """Test that half-open transitions back to open on failure."""
        import gpt_trader.features.brokerages.coinbase.client.circuit_breaker as cb_module

        current_time = [1000.0]

        class FakeTime:
            @staticmethod
            def time():
                return current_time[0]

        monkeypatch.setattr(cb_module, "time", FakeTime)

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Advance time past recovery
        current_time[0] = 1015.0
        assert breaker.state == CircuitState.HALF_OPEN

        # Any failure reopens circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_reset_returns_to_closed(self) -> None:
        """Test that reset returns breaker to closed state."""
        breaker = CircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_proceed() is True

    def test_get_status(self) -> None:
        """Test getting circuit breaker status."""
        breaker = CircuitBreaker(failure_threshold=3)

        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0

        breaker.record_failure()
        status = breaker.get_status()
        assert status["failure_count"] == 1

    def test_thread_safety(self) -> None:
        """Test that circuit breaker is thread-safe."""
        breaker = CircuitBreaker(failure_threshold=100)
        errors: list[Exception] = []

        def record_failures() -> None:
            try:
                for _ in range(50):
                    breaker.record_failure()
            except Exception as e:
                errors.append(e)

        def record_successes() -> None:
            try:
                for _ in range(50):
                    breaker.record_success()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(5):
                futures.append(executor.submit(record_failures))
                futures.append(executor.submit(record_successes))

            for f in futures:
                f.result()

        assert len(errors) == 0


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


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = CircuitOpenError("orders", 30.0)

        assert error.category == "orders"
        assert error.time_until_retry == 30.0
        assert "orders" in str(error)
        assert "30.0" in str(error)
