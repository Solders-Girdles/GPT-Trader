"""Tests for CircuitBreakerRegistry behavior and API resilience integration."""

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from gpt_trader.features.brokerages.coinbase.client.priority import (
    RequestDeferredError,
    RequestPriority,
)
from gpt_trader.features.brokerages.coinbase.client.response_cache import ResponseCache


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry class."""

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


class TestCoinbaseClientBaseRequestResilience:
    """Test CoinbaseClientBase request resilience components."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_request_uses_cache(self) -> None:
        """Test request short-circuits when cache hits."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._response_cache = ResponseCache(enabled=True)
        client._response_cache.set("/api/v3/test", {"cached": True})
        client._perform_http_request = Mock(return_value={"fresh": True})

        result = client._request("GET", "/api/v3/test")

        assert result == {"cached": True}
        client._perform_http_request.assert_not_called()

    def test_request_priority_deferred(self) -> None:
        """Test request defers when priority manager blocks."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._priority_manager = Mock()
        client._priority_manager.should_allow.return_value = False
        client._priority_manager.get_priority.return_value = RequestPriority.LOW
        client.get_rate_limit_usage = Mock(return_value=0.9)

        with pytest.raises(RequestDeferredError):
            client._request("GET", "/api/v3/products")

    def test_request_circuit_open(self) -> None:
        """Test circuit breaker blocks requests when open."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        class _Breaker:
            def get_status(self) -> dict[str, float]:
                return {"time_until_half_open": 12.0}

        class _Circuit:
            def can_proceed(self, path: str) -> bool:
                return False

            def get_breaker(self, path: str) -> _Breaker:
                return _Breaker()

            def _categorize_endpoint(self, path: str) -> str:
                return "orders"

        client._circuit_breaker = _Circuit()

        with pytest.raises(CircuitOpenError) as exc:
            client._request("GET", "/api/v3/orders")

        assert exc.value.category == "orders"

    def test_request_caches_get_response(self) -> None:
        """Test GET responses are cached when cache is enabled."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._response_cache = ResponseCache(enabled=True)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        client._request("GET", "/api/v3/test")

        assert client._response_cache.get("/api/v3/test") == {"success": True}

    def test_invalidate_cache_orders_and_positions(self) -> None:
        """Test cache invalidation targets orders and positions."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        cache = ResponseCache(enabled=True)
        cache.set("/api/v3/brokerage/orders", {"orders": [{"id": "1"}]})
        cache.set("/api/v3/brokerage/fills", {"fills": []})
        cache.set("/api/v3/brokerage/accounts", {"accounts": []})
        cache.set("/api/v3/brokerage/positions", {"positions": []})
        client._response_cache = cache

        client._invalidate_cache("POST", "/api/v3/brokerage/orders")
        assert cache.get("/api/v3/brokerage/orders") is None
        assert cache.get("/api/v3/brokerage/fills") is None
        assert cache.get("/api/v3/brokerage/accounts") is not None

        client._invalidate_cache("DELETE", "/api/v3/brokerage/positions")
        assert cache.get("/api/v3/brokerage/accounts") is None
        assert cache.get("/api/v3/brokerage/positions") is None
