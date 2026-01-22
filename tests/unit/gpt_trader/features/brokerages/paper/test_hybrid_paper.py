"""Tests for HybridPaperBroker initialization and status."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

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
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture."""
        return broker_factory(initial_equity=Decimal("10000"))

    def test_is_connected_always_true(self, broker: HybridPaperBroker) -> None:
        """Test is_connected returns True."""
        assert broker.is_connected() is True

    def test_is_stale_always_false(self, broker: HybridPaperBroker) -> None:
        """Test is_stale returns False."""
        assert broker.is_stale("BTC-USD") is False

    def test_get_status_returns_status(self, broker: HybridPaperBroker) -> None:
        """Test get_status returns status dict."""
        result = broker.get_status()

        assert result["mode"] == "paper"
        assert result["initial_equity"] == 10000.0
        assert result["current_equity"] == 10000.0
        assert result["positions"] == 0
        assert result["orders_executed"] == 0
