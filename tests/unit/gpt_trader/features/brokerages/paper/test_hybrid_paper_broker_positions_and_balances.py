"""Tests for `HybridPaperBroker` position and balance methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.paper.hybrid as hybrid_module
from gpt_trader.core import Balance, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerPositionsBalances:
    """Test HybridPaperBroker position and balance methods."""

    @pytest.fixture
    def broker(self, monkeypatch: pytest.MonkeyPatch):
        """Create broker fixture with mocked client."""
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", Mock())
        monkeypatch.setattr(hybrid_module, "SimpleAuth", Mock())
        return HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("10000"),
        )

    def test_list_positions_empty(self, broker) -> None:
        """Test list_positions returns empty list initially."""
        result = broker.list_positions()

        assert result == []

    def test_list_positions_returns_positions(self, broker) -> None:
        """Test list_positions returns stored positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.list_positions()

        assert len(result) == 1
        assert result[0].symbol == "BTC-USD"

    def test_get_positions_alias(self, broker) -> None:
        """Test get_positions is alias for list_positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.get_positions()

        assert len(result) == 1

    def test_list_balances_returns_balances(self, broker) -> None:
        """Test list_balances returns balances."""
        result = broker.list_balances()

        assert len(result) == 1
        assert result[0].asset == "USD"
        assert result[0].total == Decimal("10000")

    def test_get_balances_alias(self, broker) -> None:
        """Test get_balances is alias for list_balances."""
        result = broker.get_balances()

        assert len(result) == 1

    def test_get_equity_cash_only(self, broker) -> None:
        """Test get_equity with cash only."""
        result = broker.get_equity()

        assert result == Decimal("10000")

    def test_get_equity_with_long_position(self, broker) -> None:
        """Test get_equity includes unrealized PnL from long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("51000")  # Price went up
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("5000"), available=Decimal("5000")
        )

        result = broker.get_equity()

        # 5000 cash + (51000 - 50000) * 0.1 = 5000 + 100 = 5100
        assert result == Decimal("5100")

    def test_get_equity_with_short_position(self, broker) -> None:
        """Test get_equity includes unrealized PnL from short position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("-0.1"),  # Short
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("49000")  # Price went down (profit for short)
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("15000"), available=Decimal("15000")
        )

        result = broker.get_equity()

        # 15000 cash + (50000 - 49000) * 0.1 = 15000 + 100 = 15100
        assert result == Decimal("15100")
