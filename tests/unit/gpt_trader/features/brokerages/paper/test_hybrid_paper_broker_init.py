"""Tests for `HybridPaperBroker` initialization."""

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
