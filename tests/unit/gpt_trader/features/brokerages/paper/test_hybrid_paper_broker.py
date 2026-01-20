"""Tests for HybridPaperBroker initialization and status methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.features.brokerages.paper.hybrid as hybrid_module
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerInit:
    """Test HybridPaperBroker initialization."""

    @pytest.fixture
    def auth_mock(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock_auth = MagicMock()
        monkeypatch.setattr(hybrid_module, "SimpleAuth", mock_auth)
        return mock_auth

    @pytest.fixture
    def client_mock(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock_client = MagicMock()
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", mock_client)
        return mock_client

    def test_init_creates_client(self, auth_mock: MagicMock, client_mock: MagicMock) -> None:
        """Test initialization creates Coinbase client."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
        )

        auth_mock.assert_called_once_with(key_name="test_key", private_key="test_private_key")
        client_mock.assert_called_once()
        assert broker._initial_equity == Decimal("10000")
        assert broker._slippage_bps == 5
        assert broker._commission_bps == Decimal("5")

    def test_init_with_custom_parameters(
        self, auth_mock: MagicMock, client_mock: MagicMock
    ) -> None:
        """Test initialization with custom parameters."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("50000"),
            slippage_bps=10,
            commission_bps=Decimal("10"),
        )

        assert broker._initial_equity == Decimal("50000")
        assert broker._slippage_bps == 10
        assert broker._commission_bps == Decimal("10")

    def test_init_creates_usd_balance(self, auth_mock: MagicMock, client_mock: MagicMock) -> None:
        """Test initialization creates USD balance."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("25000"),
        )

        assert "USD" in broker._balances
        assert broker._balances["USD"].total == Decimal("25000")
        assert broker._balances["USD"].available == Decimal("25000")


class TestHybridPaperBrokerStatus:
    """Test HybridPaperBroker status methods."""

    @pytest.fixture
    def broker(self, monkeypatch: pytest.MonkeyPatch) -> HybridPaperBroker:
        """Create broker fixture."""
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", Mock())
        monkeypatch.setattr(hybrid_module, "SimpleAuth", Mock())
        return HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("10000"),
        )

    def test_is_connected_always_true(self, broker: HybridPaperBroker) -> None:
        """Test is_connected returns True."""
        assert broker.is_connected() is True

    def test_is_stale_always_false(self, broker: HybridPaperBroker) -> None:
        """Test is_stale returns False."""
        assert broker.is_stale("BTC-USD") is False

    def test_start_market_data_prefetches_quotes(self, broker: HybridPaperBroker) -> None:
        """Test start_market_data prefetches quotes."""
        broker._client = Mock()
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "50000",
            "best_ask": "50100",
            "trades": [{"price": "50050"}],
        }

        broker.start_market_data(["BTC-USD", "ETH-USD"])

        assert broker._client.get_market_product_ticker.call_count == 2

    def test_stop_market_data_noop(self, broker: HybridPaperBroker) -> None:
        """Test stop_market_data is no-op."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        result = broker.stop_market_data()

        assert result is None
        assert broker._last_prices["BTC-USD"] == Decimal("50000")

    def test_get_status_returns_status(self, broker: HybridPaperBroker) -> None:
        """Test get_status returns status dict."""
        result = broker.get_status()

        assert result["mode"] == "paper"
        assert result["initial_equity"] == 10000.0
        assert result["current_equity"] == 10000.0
        assert result["positions"] == 0
        assert result["orders_executed"] == 0
