"""Tests for CoinbaseClientBase retry and rate-limit behavior."""

import time
from unittest.mock import Mock

import pytest
import requests

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.errors import RateLimitError


@pytest.fixture
def sleep_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_sleep = Mock()
    monkeypatch.setattr(time, "sleep", mock_sleep)
    return mock_sleep


class TestCoinbaseClientBaseRequestRetries:
    """Test CoinbaseClientBase retry behavior."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_request_rate_limit_invalid_retry_after(self, sleep_mock: Mock) -> None:
        """Test rate limit retry-after defaults on invalid header."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            (429, {"retry-after": "nope"}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        sleep_mock.assert_called_once_with(1.0)

    def test_request_rate_limit_exhausts_retries(self, sleep_mock: Mock) -> None:
        """Test rate limit errors exhaust retries and raise."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (429, {"retry-after": "0"}, '{"error": "rate_limited"}')
        client.set_transport_for_testing(mock_transport)

        with pytest.raises(RateLimitError):
            client._request("GET", "/api/v3/test")

    def test_request_5xx_error_with_retry(self, sleep_mock: Mock) -> None:
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

        result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        sleep_mock.assert_called_once()
        assert client.get_api_metrics()["total_errors"] == 0

    def test_request_rate_limit_error_with_retry_after(self, sleep_mock: Mock) -> None:
        """Test request with rate limit error and retry-after header."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.side_effect = [
            (429, {"retry-after": "5"}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        sleep_mock.assert_called_once_with(5.0)  # Should use retry-after value

    def test_request_network_error_with_retry(self, sleep_mock: Mock) -> None:
        """Test request with network error and retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._metrics = APIMetricsCollector(max_history=10)
        mock_transport = Mock()

        # Requests logic catches requests.ConnectionError.
        # But when using transport, the perform_request calls transport.
        # The retry loop catches (requests.ConnectionError, requests.Timeout).
        # So transport needs to raise requests.ConnectionError.
        mock_transport.side_effect = [
            requests.ConnectionError("Network error"),
            (200, {}, '{"success": true}'),
        ]
        client.set_transport_for_testing(mock_transport)

        result = client._request("GET", "/api/v3/test")

        assert result == {"success": True}
        assert mock_transport.call_count == 2
        sleep_mock.assert_called_once()
        assert client.get_api_metrics()["total_errors"] == 0

    def test_request_max_retries_exceeded(self, sleep_mock: Mock) -> None:
        """Test request with max retries exceeded."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (500, {}, '{"error": "INTERNAL_ERROR"}')
        client.set_transport_for_testing(mock_transport)

        with pytest.raises(Exception):  # Should raise some kind of HTTP error
            client._request("GET", "/api/v3/test")

        # Should have attempted max retries + 1 initial call
        assert mock_transport.call_count == 4  # Default max_retries is 3
