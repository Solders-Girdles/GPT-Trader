"""Tests for CoinbaseClientBase request happy-path behavior."""

import json
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseRequestHappyPath:
    """Test CoinbaseClientBase request success paths."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

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

    def test_request_with_correlation_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test request with correlation ID."""
        monkeypatch.setattr(base_module, "get_correlation_id", lambda: "test-correlation-123")
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        mock_transport = Mock()
        mock_transport.return_value = (200, {}, '{"success": true}')
        client.set_transport_for_testing(mock_transport)

        client._request("GET", "/api/v3/test")

        call_args = mock_transport.call_args
        headers = call_args[0][2]
        assert headers["X-Correlation-Id"] == "test-correlation-123"
