"""Tests for `HybridPaperBroker` initialization."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerInit:
    """Test HybridPaperBroker initialization."""

    @patch("gpt_trader.features.brokerages.paper.hybrid.CoinbaseClient")
    @patch("gpt_trader.features.brokerages.paper.hybrid.SimpleAuth")
    def test_init_creates_client(self, mock_auth, mock_client) -> None:
        """Test initialization creates Coinbase client."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
        )

        mock_auth.assert_called_once_with(key_name="test_key", private_key="test_private_key")
        mock_client.assert_called_once()
        assert broker._initial_equity == Decimal("10000")
        assert broker._slippage_bps == 5
        assert broker._commission_bps == Decimal("5")

    @patch("gpt_trader.features.brokerages.paper.hybrid.CoinbaseClient")
    @patch("gpt_trader.features.brokerages.paper.hybrid.SimpleAuth")
    def test_init_with_custom_parameters(self, mock_auth, mock_client) -> None:
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

    @patch("gpt_trader.features.brokerages.paper.hybrid.CoinbaseClient")
    @patch("gpt_trader.features.brokerages.paper.hybrid.SimpleAuth")
    def test_init_creates_usd_balance(self, mock_auth, mock_client) -> None:
        """Test initialization creates USD balance."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("25000"),
        )

        assert "USD" in broker._balances
        assert broker._balances["USD"].total == Decimal("25000")
        assert broker._balances["USD"].available == Decimal("25000")
