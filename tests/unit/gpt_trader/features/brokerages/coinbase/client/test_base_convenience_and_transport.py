"""Tests for CoinbaseClientBase convenience methods and transport behavior."""

from unittest.mock import Mock

import requests

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseConvenienceAndTransport:
    """Test CoinbaseClientBase convenience methods and session transport."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

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
