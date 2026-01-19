"""Tests for CoinbaseClientBase endpoint/path helpers."""

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError


class TestCoinbaseClientBaseEndpointsAndPaths:
    """Test CoinbaseClientBase endpoint and URL helpers."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

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
