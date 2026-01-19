"""Tests for CoinbaseClientBase client error handling and response parsing."""

from unittest.mock import Mock

import pytest
import requests

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.errors import (
    InvalidRequestError,
    PermissionDeniedError,
)


class TestCoinbaseClientBaseRequestClientErrors:
    """Test CoinbaseClientBase request client error behavior."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

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
