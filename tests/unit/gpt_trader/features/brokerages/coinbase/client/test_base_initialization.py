"""Tests for CoinbaseClientBase initialization and configuration."""

from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth, SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseInitialization:
    """Test CoinbaseClientBase initialization behavior."""

    def setup_method(self) -> None:
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
