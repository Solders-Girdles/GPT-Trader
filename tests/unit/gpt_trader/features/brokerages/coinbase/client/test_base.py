"""Tests for Coinbase client base functionality."""

import json
import time
from unittest.mock import Mock, patch

import pytest
import requests

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth, SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import CircuitOpenError
from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.client.priority import (
    RequestDeferredError,
    RequestPriority,
)
from gpt_trader.features.brokerages.coinbase.client.response_cache import ResponseCache
from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    InvalidRequestError,
    PermissionDeniedError,
    RateLimitError,
)


class TestCoinbaseClientBase:
    """Test CoinbaseClientBase class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_client_init_default_params(self) -> None:
        """Test client initialization with default parameters."""
        client = CoinbaseClientBase(
            base_url=self.base_url,
            auth=self.auth,
        )

        assert client.base_url == self.base_url
        assert client.auth == self.auth
        assert client.timeout == 30
        assert client.api_version == "2024-10-24"
        assert client.rate_limit_per_minute == 100
        assert client.enable_throttle is True
        assert client.api_mode == "advanced"
        assert client.enable_keep_alive is True
        assert client._is_cdp is False

    def test_client_init_custom_params(self) -> None:
        """Test client initialization with custom parameters."""
        cdp_auth = Mock(spec=CDPJWTAuth)
        cdp_auth.key_name = "organizations/test-org"
        client = CoinbaseClientBase(
            base_url=self.base_url,
            auth=cdp_auth,
            timeout=60,
            api_version="2023-01-01",
            rate_limit_per_minute=200,
            enable_throttle=False,
            api_mode="exchange",
            enable_keep_alive=False,
        )

        assert client.timeout == 60
        assert client.api_version == "2023-01-01"
        assert client.rate_limit_per_minute == 200
        assert client.enable_throttle is False
        assert client.api_mode == "exchange"
        assert client.enable_keep_alive is False
        assert client._is_cdp is True

    def test_client_init_no_auth(self) -> None:
        """Test client initialization without authentication."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=None)

        assert client.auth is None
        assert client._is_cdp is False

    def test_client_init_disables_resilience_components(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test client initialization with resilience flags disabled."""
        monkeypatch.setattr(base_module, "CACHE_ENABLED", False)
        monkeypatch.setattr(base_module, "CIRCUIT_BREAKER_ENABLED", False)
        monkeypatch.setattr(base_module, "METRICS_ENABLED", False)
        monkeypatch.setattr(base_module, "PRIORITY_ENABLED", False)

        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        assert client._response_cache is None
        assert client._circuit_breaker is None
        assert client._metrics is None
        assert client._priority_manager is None

    def test_set_transport_for_testing(self) -> None:
        """Test setting custom transport for testing."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_keep_alive=False)
        mock_transport = Mock()

        client.set_transport_for_testing(mock_transport)

        assert client._transport == mock_transport

    def test_get_endpoint_path_advanced_mode(self) -> None:
        """Test endpoint path resolution for advanced API mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        # Test basic endpoints
        assert client._get_endpoint_path("products") == "/api/v3/brokerage/products"
        assert client._get_endpoint_path("accounts") == "/api/v3/brokerage/accounts"

        # Test endpoints with parameters
        path = client._get_endpoint_path("product", product_id="BTC-USD")
        assert path == "/api/v3/brokerage/products/BTC-USD"

        path = client._get_endpoint_path("account", account_uuid="123-456")
        assert path == "/api/v3/brokerage/accounts/123-456"

    def test_get_endpoint_path_exchange_mode(self) -> None:
        """Test endpoint path resolution for exchange API mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="exchange")

        assert client._get_endpoint_path("products") == "/products"
        assert client._get_endpoint_path("accounts") == "/accounts"

        path = client._get_endpoint_path("product", product_id="BTC-USD")
        assert path == "/products/BTC-USD"

    def test_get_endpoint_path_missing_param_returns_template(self) -> None:
        """Test endpoint path resolution when params are missing."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        path = client._get_endpoint_path("product")

        assert "{product_id}" in path

    def test_get_endpoint_path_invalid_mode(self) -> None:
        """Test endpoint path resolution with invalid API mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="invalid")

        with pytest.raises(InvalidRequestError, match="Unknown API mode: invalid"):
            client._get_endpoint_path("products")

    def test_get_endpoint_path_unknown_endpoint(self) -> None:
        """Test endpoint path resolution with unknown endpoint."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        with pytest.raises(InvalidRequestError, match="Unknown endpoint: unknown_endpoint"):
            client._get_endpoint_path("unknown_endpoint")

    def test_get_endpoint_path_wrong_mode(self) -> None:
        """Test endpoint path resolution when endpoint not available in current mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="exchange")

        # Some endpoints are only available in advanced mode, e.g. orders
        # In exchange mode, "orders" key is not in map.
        # The code checks if it exists in advanced but not current.

        # We use "orders" which is in advanced but not exchange
        with pytest.raises(InvalidRequestError, match="not available in exchange mode"):
            client._get_endpoint_path("orders")

    def test_make_url_with_full_url(self) -> None:
        """Test URL making with full URL."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_keep_alive=False)
        full_url = "https://api.example.com/test"

        result = client._make_url(full_url)

        assert result == full_url

    def test_make_url_with_path(self) -> None:
        """Test URL making with path."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        result = client._make_url("/api/v3/test")

        assert result == "https://api.coinbase.com/api/v3/test"

        result = client._make_url("api/v3/test")

        assert result == "https://api.coinbase.com/api/v3/test"

    def test_build_path_with_params(self) -> None:
        """Test building path with query parameters."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        # Test with no params
        result = client._build_path_with_params("/api/v3/test", None)
        assert result == "/api/v3/test"

        result = client._build_path_with_params("/api/v3/test", {})
        assert result == "/api/v3/test"

        result = client._build_path_with_params("/api/v3/test", {"limit": None, "offset": None})
        assert result == "/api/v3/test"

        # Test with params
        result = client._build_path_with_params("/api/v3/test", {"limit": "100", "offset": "0"})
        assert result == "/api/v3/test?limit=100&offset=0"

        # Test with existing query string
        result = client._build_path_with_params("/api/v3/test?status=active", {"limit": "100"})
        assert result == "/api/v3/test?status=active&limit=100"

    def test_normalize_path(self) -> None:
        """Test path normalization."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        # Test with full URL
        result = client._normalize_path("https://api.coinbase.com/api/v3/test")
        assert result == "/api/v3/test"

        # Test with path
        result = client._normalize_path("/api/v3/test")
        assert result == "/api/v3/test"

        result = client._normalize_path("api/v3/test")
        assert result == "api/v3/test"

    def test_build_auth_path_strips_query(self) -> None:
        """Test auth path normalization strips queries and full URLs."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        auth_path = client._build_auth_path(
            "https://api.coinbase.com/api/v3/brokerage/orders?limit=10"
        )
        assert auth_path == "/api/v3/brokerage/orders"

        auth_path = client._build_auth_path("api/v3/brokerage/orders?limit=5")
        assert auth_path == "/api/v3/brokerage/orders"

    def test_public_market_endpoint_skips_auth(self) -> None:
        """Test public market endpoints do not include auth headers."""
        auth = Mock(spec=SimpleAuth)
        auth.get_headers.return_value = {"Authorization": "Bearer token"}
        client = CoinbaseClientBase(base_url=self.base_url, auth=auth, api_mode="advanced")

        headers = client._build_headers("GET", "/api/v3/brokerage/market/products", None)

        auth.get_headers.assert_not_called()
        assert "Authorization" not in headers

    def test_is_public_market_endpoint_exchange_mode(self) -> None:
        """Test public endpoint detection ignores non-advanced mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="exchange")

        assert client._is_public_market_endpoint("/api/v3/brokerage/market/products") is False

    def test_is_public_market_endpoint_normalizes_missing_slash(self) -> None:
        """Test public endpoint detection normalizes missing leading slash."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        assert client._is_public_market_endpoint("api/v3/brokerage/market/products") is True

    def test_apply_auth_headers_strips_query(self) -> None:
        """Test auth headers use normalized path without query params."""
        auth = Mock(spec=SimpleAuth)
        auth.get_headers.return_value = {"Authorization": "Bearer token"}
        client = CoinbaseClientBase(base_url=self.base_url, auth=auth)
        headers: dict[str, str] = {}

        client._apply_auth_headers(headers, "GET", "/api/v3/test?limit=1", None)

        auth.get_headers.assert_called_once_with("GET", "/api/v3/test")

    def test_apply_auth_headers_uses_sign(self) -> None:
        """Test auth headers use sign() when available."""

        class SignAuth:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, dict | None]] = []

            def sign(self, method: str, path: str, payload: dict | None) -> dict[str, str]:
                self.calls.append((method, path, payload))
                return {"Authorization": "Signed token"}

        auth = SignAuth()
        client = CoinbaseClientBase(base_url=self.base_url, auth=auth)
        headers: dict[str, str] = {}

        client._apply_auth_headers(headers, "POST", "/api/v3/test", {"x": 1})

        assert headers["Authorization"] == "Signed token"
        assert auth.calls == [("POST", "/api/v3/test", {"x": 1})]

    def test_apply_auth_headers_no_auth(self) -> None:
        """Test auth header helper no-ops without auth."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=None)
        headers = {"X-Test": "1"}

        client._apply_auth_headers(headers, "GET", "/api/v3/test", None)

        assert headers == {"X-Test": "1"}

    def test_convenience_methods(self) -> None:
        """Test convenience HTTP methods."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._request = Mock(return_value={"success": True})

        # Test GET
        result = client.get("/api/v3/test", {"param": "value"})
        client._request.assert_called_once_with("GET", "/api/v3/test?param=value")
        assert result == {"success": True}

        client._request.reset_mock()

        # Test POST
        result = client.post("/api/v3/test", {"data": "value"})
        client._request.assert_called_once_with("POST", "/api/v3/test", {"data": "value"})
        assert result == {"success": True}

        client._request.reset_mock()

        # Test DELETE
        result = client.delete("/api/v3/test", {"data": "value"})
        client._request.assert_called_once_with("DELETE", "/api/v3/test", {"data": "value"})
        assert result == {"success": True}

    def test_perform_request_uses_session(self) -> None:
        """Test request goes through session when no transport is set."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        response = requests.Response()
        response.status_code = 200
        client.session.request = Mock(return_value=response)

        headers = {"X-Test": "1"}
        result = client._perform_request("GET", "https://api.coinbase.com/test", headers, None)

        assert result is response
        client.session.request.assert_called_once_with(
            "GET",
            "https://api.coinbase.com/test",
            json=None,
            headers=headers,
            timeout=client.timeout,
        )

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    def test_check_rate_limit_disabled(self, mock_time: Mock) -> None:
        """Test rate limit checking when disabled."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_throttle=False)

        # Should not raise any exceptions or sleep
        client._check_rate_limit()

        assert len(client._request_times) == 0

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_normal_usage(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit checking with normal usage."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add some requests within the limit
        for i in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5
        mock_sleep.assert_not_called()

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_warning_threshold(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit warning threshold."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add requests up to 80% of limit
        # Need to trigger the warning, which happens when len >= 8
        # So we need 8 items in the list, then the next call (9th) warns (no wait, check is first)
        # Check is: if len >= limit * 0.8
        # So if len is 8, it warns.
        # So we need to call it 8 times to populate list to 8.
        # Then call it once more to trigger warning.

        for i in range(9):
            client._check_rate_limit()

        mock_logger.warning.assert_called_once_with(
            "Approaching rate limit: %d/%d requests in last minute",
            8,
            10,
        )

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_exceeded(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit exceeded behavior."""
        # Start at time 1000
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=5)

        # Add requests up to the limit
        for i in range(5):
            client._check_rate_limit()

        # Next request should trigger rate limiting
        # Advance time by 30 seconds (still within 1-minute window)
        mock_time.return_value = 1030.0
        client._check_rate_limit()

        # Should have slept to respect rate limit
        assert mock_sleep.call_count >= 1
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        assert any(s > 30 for s in sleep_calls)  # remaining time + buffer

        mock_logger.info.assert_called_once()
        assert "Rate limit reached" in mock_logger.info.call_args[0][0]

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    def test_check_rate_limit_window_cleanup(self, mock_time: Mock) -> None:
        """Test rate limit window cleanup."""
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add requests at time 1000
        mock_time.return_value = 1000.0
        for i in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5

        # Advance time beyond 1 minute window
        mock_time.return_value = 2000.0
        client._check_rate_limit()

        # Old requests should be cleaned up
        assert len(client._request_times) == 1
        assert client._request_times[0] == 2000.0

    def test_request_success(self) -> None:
        """Test successful request."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        mock_transport.assert_called_once()

        # Check that auth was used
        self.auth.get_headers.assert_called_once_with("GET", "/api/v3/test")

        # Check headers
        call_args = mock_transport.call_args
        headers = call_args[0][2]
        assert headers["Content-Type"] == "application/json"
        assert headers["CB-VERSION"] == "2024-10-24"
        assert "Authorization" in headers

    def test_request_with_body(self) -> None:
        """Test request with body."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        body = {"test": "data"}
        result = client._request("POST", "/api/v3/test", body)

        assert result == {"success": True}

        # Check that body was passed and auth was called with it
        call_args = mock_transport.call_args
        request_body = call_args[0][3]

        # request_body is a string (json.dumps), not bytes
        assert json.loads(request_body) == body

        self.auth.get_headers.assert_called_once_with("POST", "/api/v3/test", body)

    def test_request_no_auth(self) -> None:
        """Test request without authentication."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=None)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        client._request("GET", "/api/v3/test")

        # Should not attempt to sign
        call_args = mock_transport.call_args
        headers = call_args[0][2]
        assert "Authorization" not in headers

    @patch("gpt_trader.features.brokerages.coinbase.client.base.get_correlation_id")
    def test_request_with_correlation_id(self, mock_get_correlation_id: Mock) -> None:
        """Test request with correlation ID."""
        mock_get_correlation_id.return_value = "test-correlation-123"
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        client._request("GET", "/api/v3/test")

        call_args = mock_transport.call_args
        headers = call_args[0][2]
        assert headers["X-Correlation-Id"] == "test-correlation-123"

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

    def test_request_4xx_error(self) -> None:
        """Test request with 4xx error."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (
            400,
            {},
            '{"error": "INVALID_REQUEST", "message": "Bad request"}',
        )
        client.set_transport_for_testing(mock_transport)

        with pytest.raises(InvalidRequestError, match="Bad request"):
            client._request("GET", "/api/v3/test")

    def test_request_4xx_invalid_json_uses_text(self) -> None:
        """Test 4xx errors fall back to raw response text."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (400, {}, "not-json")
        client.set_transport_for_testing(mock_transport)

        with pytest.raises(InvalidRequestError, match="not-json"):
            client._request("GET", "/api/v3/test")

    def test_raise_client_error_forbidden(self) -> None:
        """Test non-400 errors map to appropriate client errors."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        resp = requests.Response()
        resp.status_code = 403
        resp._content = b'{"message": "forbidden", "error": "FORBIDDEN"}'

        with pytest.raises(PermissionDeniedError, match="forbidden"):
            client._raise_client_error(resp)

    def test_request_rate_limit_invalid_retry_after(self) -> None:
        """Test rate limit retry-after defaults on invalid header."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            (429, {"retry-after": "nope"}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep") as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        mock_sleep.assert_called_once_with(1.0)

    def test_request_rate_limit_exhausts_retries(self) -> None:
        """Test rate limit errors exhaust retries and raise."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (429, {"retry-after": "0"}, '{"error": "rate_limited"}')
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                client._request("GET", "/api/v3/test")

    def test_request_5xx_error_with_retry(self) -> None:
        """Test request with 5xx error and retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._metrics = APIMetricsCollector(max_history=10)
        mock_transport = Mock()
        # First call fails, second succeeds
        mock_transport.side_effect = [
            (500, {}, '{"error": "INTERNAL_ERROR", "message": "Server error"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep") as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        mock_sleep.assert_called_once()
        assert client.get_api_metrics()["total_errors"] == 0

    def test_request_rate_limit_error_with_retry_after(self) -> None:
        """Test request with rate limit error and retry-after header."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            (429, {"retry-after": "5"}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep") as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        mock_sleep.assert_called_once_with(5.0)  # Should use retry-after value

    def test_request_network_error_with_retry(self) -> None:
        """Test request with network error and retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._metrics = APIMetricsCollector(max_history=10)
        mock_transport = Mock()

        # Requests logic catches requests.ConnectionError.
        # But when using transport, the perform_request calls transport.
        # The retry loop catches (requests.ConnectionError, requests.Timeout).
        # So transport needs to raise requests.ConnectionError.

        import requests

        mock_transport.side_effect = [
            requests.ConnectionError("Network error"),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep") as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        mock_sleep.assert_called_once()
        assert client.get_api_metrics()["total_errors"] == 0

    def test_request_max_retries_exceeded(self) -> None:
        """Test request with max retries exceeded."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (500, {}, '{"error": "INTERNAL_ERROR"}')
        client.set_transport_for_testing(mock_transport)

        with patch("time.sleep"):
            with pytest.raises(Exception):  # Should raise some kind of HTTP error
                client._request("GET", "/api/v3/test")

        # Should have attempted max retries + 1 initial call
        assert mock_transport.call_count == 4  # Default max_retries is 3

    def test_request_invalid_json_response(self) -> None:
        """Test request with invalid JSON response."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, "invalid json response")
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"raw": "invalid json response"}

    def test_request_empty_response(self) -> None:
        """Test request with empty response."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, "")
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {}

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

    def test_record_success_sets_span_and_breaker(self) -> None:
        """Test success recording updates circuit breaker and span."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._circuit_breaker = Mock()
        resp = requests.Response()
        resp.status_code = 200
        span = Mock()

        client._record_success("/api/v3/test", resp, span)

        client._circuit_breaker.record_success.assert_called_once_with("/api/v3/test")
        span.set_attribute.assert_called_once_with("http.status_code", 200)

    def test_record_request_metrics_records_metrics_and_span(self) -> None:
        """Test request metrics recording updates metrics and span attributes."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._metrics = Mock()
        span = Mock()

        with (
            patch(
                "gpt_trader.features.brokerages.coinbase.client.base.record_histogram"
            ) as mock_hist,
            patch(
                "gpt_trader.features.brokerages.coinbase.client.base.record_counter"
            ) as mock_counter,
        ):
            client._record_request_metrics("/api/v3/test", 0.25, True, False, span)

        client._metrics.record_request.assert_called_once_with(
            "/api/v3/test",
            250.0,
            error=True,
            rate_limited=False,
        )
        mock_hist.assert_called_once()
        mock_counter.assert_called_once()
        span.set_attribute.assert_any_call("http.latency_ms", 250.0)
        span.set_attribute.assert_any_call("http.rate_limited", False)

    def test_paginate_success(self) -> None:
        """Test successful pagination."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        # Mock responses for multiple pages
        page1 = {"data": [{"id": 1}, {"id": 2}], "cursor": "page2"}
        page2 = {"data": [{"id": 3}], "cursor": None}

        client._request = Mock(side_effect=[page1, page2])

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 3
        assert results == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert client._request.call_count == 2

        # Check that cursor was passed in second request
        second_call_args = client._request.call_args_list[1]
        assert "cursor=page2" in second_call_args[0][1]

    def test_paginate_custom_cursor_params(self) -> None:
        """Test pagination with custom cursor parameters."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        pages = [
            {"items": [{"id": 1}], "next_cursor": "next"},
            {"items": [], "next_cursor": None},
        ]
        client._request = Mock(side_effect=pages)

        list(
            client.paginate(
                "/api/v3/test",
                {"limit": "100"},
                "items",
                cursor_param="page_token",
                cursor_field="next_cursor",
            )
        )

        # Check that custom cursor param was used
        call_args = client._request.call_args
        assert "page_token=next" in call_args[0][1]

    def test_paginate_with_pagination_object(self) -> None:
        """Test pagination with nested pagination cursor."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page1 = {"items": [{"id": 1}], "pagination": {"next_cursor": "next"}}
        page2 = {"items": [], "pagination": {"next_cursor": None}}
        client._request = Mock(side_effect=[page1, page2])

        list(client.paginate("/api/v3/test", {}, "items"))

        second_call_args = client._request.call_args_list[1]
        assert "cursor=next" in second_call_args[0][1]

    def test_paginate_no_cursor(self) -> None:
        """Test pagination when no cursor is returned."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page = {"data": [{"id": 1}]}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 1
        assert client._request.call_count == 1

    def test_paginate_non_list_item_yields_single(self) -> None:
        """Test pagination yields single non-list item."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._request = Mock(return_value={"id": 1})

        results = list(client.paginate("/api/v3/test"))

        assert results == [{"id": 1}]
        assert client._request.call_count == 1

    def test_paginate_non_dict_response_stops(self) -> None:
        """Test pagination stops when response is not a dict."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._request = Mock(return_value="payload")

        results = list(client.paginate("/api/v3/test"))

        assert results == ["payload"]
        assert client._request.call_count == 1

    def test_paginate_empty_items(self) -> None:
        """Test pagination when items are empty."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page = {"data": [], "cursor": None}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 0
        # Should still perform the initial request even with no items
        assert client._request.call_count == 1

    def test_perform_http_request_http_error_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test HTTPError path when error response is not JSON."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        response = requests.Response()
        response.status_code = 500
        response._content = b"not-json"
        client._perform_request = Mock(return_value=response)
        client._check_rate_limit = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(BrokerageError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                None,
            )

    def test_perform_http_request_network_error_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test final network error attempt raises without retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._perform_request = Mock(side_effect=requests.ConnectionError("boom"))
        client._check_rate_limit = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(requests.ConnectionError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                None,
            )

    def test_perform_http_request_records_failure_and_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test failures record circuit breaker and span attributes."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._perform_request = Mock(side_effect=RuntimeError("boom"))
        client._check_rate_limit = Mock()
        client._circuit_breaker = Mock()
        span = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(RuntimeError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                span,
            )

        client._circuit_breaker.record_failure.assert_called_once()
        span.set_attribute.assert_any_call("error", True)
        span.set_attribute.assert_any_call("error.type", "RuntimeError")

    def test_get_resilience_status_records_metrics(self) -> None:
        """Test resilience status includes metrics and circuit breaker info."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.get_rate_limit_usage = Mock(return_value=0.5)
        client._metrics = Mock()
        client._metrics.get_summary.return_value = {"error_rate": 0.2}
        client._circuit_breaker = Mock()
        client._circuit_breaker.get_all_status.return_value = {"orders": {"state": "open"}}
        client._response_cache = Mock()
        client._response_cache.get_stats.return_value = {"entries": 1}
        client._priority_manager = Mock()
        client._priority_manager.get_stats.return_value = {"deferred": 2}

        with patch(
            "gpt_trader.features.brokerages.coinbase.client.base.record_gauge"
        ) as mock_gauge:
            status = client.get_resilience_status()

        assert status["metrics"] == {"error_rate": 0.2}
        assert status["circuit_breakers"] == {"orders": {"state": "open"}}
        assert status["cache"] == {"entries": 1}
        assert status["priority"] == {"deferred": 2}
        mock_gauge.assert_any_call("gpt_trader_rate_limit_usage_ratio", 0.5)
        mock_gauge.assert_any_call("gpt_trader_api_error_rate", 0.2)
        mock_gauge.assert_any_call(
            "gpt_trader_circuit_breaker_state",
            2.0,
            labels={"category": "orders"},
        )

    def test_context_manager_closes(self) -> None:
        """Test context manager closes session."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.close = Mock()

        with client as ctx:
            assert ctx is client

        client.close.assert_called_once()

    def test_close_handles_session_error(self) -> None:
        """Test close logs warning on session close error."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.session.close = Mock(side_effect=RuntimeError("boom"))

        with patch("gpt_trader.features.brokerages.coinbase.client.base.logger") as mock_logger:
            client.close()

        mock_logger.warning.assert_called_once()

    def test_del_suppresses_close_errors(self) -> None:
        """Test destructor suppresses close errors."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.close = Mock(side_effect=RuntimeError("boom"))

        client.__del__()
