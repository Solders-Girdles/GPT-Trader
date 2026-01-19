"""Tests for CoinbaseClientBase auth header helpers and public endpoints."""

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseAuthHeaders:
    """Test CoinbaseClientBase auth behaviors."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

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
