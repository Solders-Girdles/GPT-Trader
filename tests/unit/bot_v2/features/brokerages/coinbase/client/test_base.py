"""Tests for Coinbase client base functionality."""

import json
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.auth import CoinbaseAuth, CDPJWTAuth
from bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError, RateLimitError


class TestCoinbaseClientBase:
    """Test CoinbaseClientBase class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=CoinbaseAuth)
        self.auth.sign.return_value = {"Authorization": "Bearer test-token"}

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

    def test_set_transport_for_testing(self) -> None:
        """Test setting custom transport for testing."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()

        client.set_transport_for_testing(mock_transport)

        assert client._transport == mock_transport

    def test_get_endpoint_path_advanced_mode(self) -> None:
        """Test endpoint path resolution for advanced API mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        # Test basic endpoints
        assert client._get_endpoint_path("products") == "/api/v3/brokerage/market/products"
        assert client._get_endpoint_path("accounts") == "/api/v3/brokerage/accounts"

        # Test endpoints with parameters
        path = client._get_endpoint_path("product", product_id="BTC-USD")
        assert path == "/api/v3/brokerage/market/products/BTC-USD"

        path = client._get_endpoint_path("account", account_uuid="123-456")
        assert path == "/api/v3/brokerage/accounts/123-456"

    def test_get_endpoint_path_exchange_mode(self) -> None:
        """Test endpoint path resolution for exchange API mode."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="exchange")

        assert client._get_endpoint_path("products") == "/products"
        assert client._get_endpoint_path("accounts") == "/accounts"

        path = client._get_endpoint_path("product", product_id="BTC-USD")
        assert path == "/products/BTC-USD"

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
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, api_mode="advanced")

        # Some endpoints are only available in exchange mode
        with pytest.raises(InvalidRequestError, match="not available in advanced mode"):
            client._get_endpoint_path("trades")

    def test_make_url_with_full_url(self) -> None:
        """Test URL making with full URL."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
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

    @patch('bot_v2.features.brokerages.coinbase.client.base.time.time')
    def test_check_rate_limit_disabled(self, mock_time: Mock) -> None:
        """Test rate limit checking when disabled."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, enable_throttle=False
        )

        # Should not raise any exceptions or sleep
        client._check_rate_limit()

        assert len(client._request_times) == 0

    @patch('bot_v2.features.brokerages.coinbase.client.base.time.time')
    @patch('bot_v2.features.brokerages.coinbase.client.base.time.sleep')
    @patch('bot_v2.features.brokerages.coinbase.client.base.logger')
    def test_check_rate_limit_normal_usage(self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock) -> None:
        """Test rate limit checking with normal usage."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10)

        # Add some requests within the limit
        for i in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5
        mock_sleep.assert_not_called()

    @patch('bot_v2.features.brokerages.coinbase.client.base.time.time')
    @patch('bot_v2.features.brokerages.coinbase.client.base.time.sleep')
    @patch('bot_v2.features.brokerages.coinbase.client.base.logger')
    def test_check_rate_limit_warning_threshold(self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock) -> None:
        """Test rate limit warning threshold."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10)

        # Add requests up to 80% of limit
        for i in range(8):
            client._check_rate_limit()

        mock_logger.warning.assert_called_once_with(
            "Approaching rate limit: %d/%d requests in last minute",
            8,
            10,
        )

    @patch('bot_v2.features.brokerages.coinbase.client.base.time.time')
    @patch('bot_v2.features.brokerages.coinbase.client.base.time.sleep')
    @patch('bot_v2.features.brokerages.coinbase.client.base.logger')
    def test_check_rate_limit_exceeded(self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock) -> None:
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
        mock_sleep.assert_called_once()
        sleep_call = mock_sleep.call_args[0][0]
        assert sleep_call > 30  # Should sleep for remaining time + buffer

        mock_logger.info.assert_called_once()
        assert "Rate limit reached" in mock_logger.info.call_args[0][0]

    @patch('bot_v2.features.brokerages.coinbase.client.base.time.time')
    def test_check_rate_limit_window_cleanup(self, mock_time: Mock) -> None:
        """Test rate limit window cleanup."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10)

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
        self.auth.sign.assert_called_once_with("GET", "/api/v3/test", None)

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
        assert json.loads(request_body.decode()) == body

        self.auth.sign.assert_called_once_with("POST", "/api/v3/test", body)

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

    @patch('bot_v2.features.brokerages.coinbase.client.base.get_correlation_id')
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

    def test_request_4xx_error(self) -> None:
        """Test request with 4xx error."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (400, {}, '{"error": "INVALID_REQUEST", "message": "Bad request"}')
        client.set_transport_for_testing(mock_transport)

        with pytest.raises(InvalidRequestError, match="Bad request"):
            client._request("GET", "/api/v3/test")

    def test_request_5xx_error_with_retry(self) -> None:
        """Test request with 5xx error and retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        # First call fails, second succeeds
        mock_transport.side_effect = [
            (500, {}, '{"error": "INTERNAL_ERROR", "message": "Server error"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch('time.sleep') as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        mock_sleep.assert_called_once()

    def test_request_rate_limit_error_with_retry_after(self) -> None:
        """Test request with rate limit error and retry-after header."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            (429, {"retry-after": "5"}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch('time.sleep') as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        mock_sleep.assert_called_once_with(5.0)  # Should use retry-after value

    def test_request_network_error_with_retry(self) -> None:
        """Test request with network error and retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            ConnectionError("Network error"),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        with patch('time.sleep') as mock_sleep:
            result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        mock_sleep.assert_called_once()

    def test_request_max_retries_exceeded(self) -> None:
        """Test request with max retries exceeded."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (500, {}, '{"error": "INTERNAL_ERROR"}')
        client.set_transport_for_testing(mock_transport)

        with patch('time.sleep'):
            with pytest.raises(Exception):  # Should raise some kind of HTTP error
                client._request("GET", "/api/v3/test")

        # Should have attempted max retries + 1 initial call
        assert mock_transport.call_count == 4  # Default max_retries is 3

    def test_request_invalid_json_response(self) -> None:
        """Test request with invalid JSON response."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, 'invalid json response')
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"raw": "invalid json response"}

    def test_request_empty_response(self) -> None:
        """Test request with empty response."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '')
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {}

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
        assert "cursor=page2" in second_call_args[0][0]

    def test_paginate_custom_cursor_params(self) -> None:
        """Test pagination with custom cursor parameters."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        
        page = {"items": [{"id": 1}], "next_cursor": "next"}
        client._request = Mock(return_value=page)

        list(client.paginate(
            "/api/v3/test", 
            {"limit": "100"}, 
            "items", 
            cursor_param="page_token",
            cursor_field="next_cursor"
        ))

        # Check that custom cursor param was used
        call_args = client._request.call_args
        assert "page_token=next" in call_args[0][0]

    def test_paginate_no_cursor(self) -> None:
        """Test pagination when no cursor is returned."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        
        page = {"data": [{"id": 1}]}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 1
        assert client._request.call_count == 1

    def test_paginate_empty_items(self) -> None:
        """Test pagination when items are empty."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        
        page = {"data": [], "cursor": "next"}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 0
        # Should still make the request to check for next page
        assert client._request.call_count == 1

    @patch('bot_v2.features.brokerages.coinbase.client.base._load_system_config')
    def test_load_system_config_with_package_config(self, mock_load_config: Mock) -> None:
        """Test loading system config from package."""
        mock_load_config.return_value = {"max_retries": 5, "retry_delay": 2.0}
        
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (500, {}, "Server error")
        client.set_transport_for_testing(mock_transport)

        with patch('time.sleep'):
            try:
                client._request("GET", "/api/v3/test")
            except:
                pass

        mock_load_config.assert_called_once_with("system")

    def test_auth_api_mode_sync(self) -> None:
        """Test that auth API mode is synced with client API mode."""
        auth = Mock(spec=CoinbaseAuth)
        auth.api_mode = None
        
        client = CoinbaseClientBase(base_url=self.base_url, auth=auth, api_mode="exchange")
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        client._request("GET", "/api/v3/test")

        # Auth API mode should be synced
        assert auth.api_mode == "exchange"

    def test_urllib_transport_success(self) -> None:
        """Test urllib transport with successful response."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        
        with patch('bot_v2.features.brokerages.coinbase.client.base._ul') as mock_ul:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.headers.items.return_value = [("content-type", "application/json")]
            mock_response.read.return_value = b'{"success": true}'
            
            mock_urlopen = Mock()
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_ul.urlopen.return_value = mock_urlopen
            
            status, headers, text = client._urllib_transport(
                "GET", "https://api.coinbase.com/test", {}, None, 30
            )
            
            assert status == 200
            assert headers["content-type"] == "application/json"
            assert text == '{"success": true}'

    def test_urllib_transport_http_error(self) -> None:
        """Test urllib transport with HTTP error."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        
        with patch('bot_v2.features.brokerages.coinbase.client.base._ul') as mock_ul:
            from bot_v2.features.brokerages.coinbase.client import _ue
            
            mock_error = Mock()
            mock_error.code = 404
            mock_error.headers = {"content-type": "application/json"}
            mock_error.read.return_value = b'{"error": "NOT_FOUND"}'
            
            mock_urlopen = Mock(side_effect=_ue.HTTPError(
                url="https://api.coinbase.com/test",
                code=404,
                msg="Not Found",
                hdrs={"content-type": "application/json"},
                fp=Mock()
            ))
            mock_urlopen.side_effect.read.return_value = b'{"error": "NOT_FOUND"}'
            mock_ul.urlopen.return_value = mock_urlopen
            
            status, headers, text = client._urllib_transport(
                "GET", "https://api.coinbase.com/test", {}, None, 30
            )
            
            assert status == 404
            assert text == '{"error": "NOT_FOUND"}'

    def test_urllib_transport_keep_alive(self) -> None:
        """Test urllib transport with keep-alive enabled."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_keep_alive=True)
        
        with patch('bot_v2.features.brokerages.coinbase.client.base._ul') as mock_ul:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.headers.items.return_value = []
            mock_response.read.return_value = b'{"success": true}'
            
            mock_urlopen = Mock()
            mock_urlopen.return_value.__enter__.return_value = mock_response
            client._opener = Mock()
            client._opener.open.return_value = mock_urlopen
            
            status, headers, text = client._urllib_transport(
                "GET", "https://api.coinbase.com/test", {}, None, 30
            )
            
            client._opener.open.assert_called_once()
            mock_ul.urlopen.assert_not_called()
