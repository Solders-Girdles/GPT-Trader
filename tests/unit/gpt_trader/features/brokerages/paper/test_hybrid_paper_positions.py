"""Tests for HybridPaperBroker positions, balances, and updates."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import Balance, OrderSide, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


@pytest.fixture
def broker(broker_factory) -> HybridPaperBroker:
    """Create broker fixture."""
    return broker_factory(initial_equity=Decimal("10000"))


class TestHybridPaperBrokerPositionsBalances:
    """Test HybridPaperBroker position and balance methods."""

    def test_list_positions_empty(self, broker: HybridPaperBroker) -> None:
        """Test list_positions returns empty list initially."""
        result = broker.list_positions()

        assert result == []

    def test_list_positions_returns_positions(self, broker: HybridPaperBroker) -> None:
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

    def test_get_positions_alias(self, broker: HybridPaperBroker) -> None:
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

    def test_list_balances_returns_balances(self, broker: HybridPaperBroker) -> None:
        """Test list_balances returns balances."""
        result = broker.list_balances()

        assert len(result) == 1
        assert result[0].asset == "USD"
        assert result[0].total == Decimal("10000")

    def test_get_balances_alias(self, broker: HybridPaperBroker) -> None:
        """Test get_balances is alias for list_balances."""
        result = broker.get_balances()

        assert len(result) == 1

    def test_get_equity_cash_only(self, broker: HybridPaperBroker) -> None:
        """Test get_equity with cash only."""
        result = broker.get_equity()

        assert result == Decimal("10000")

    def test_get_equity_with_long_position(self, broker: HybridPaperBroker) -> None:
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

    def test_get_equity_with_short_position(self, broker: HybridPaperBroker) -> None:
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


class TestHybridPaperBrokerPositionUpdates:
    """Test HybridPaperBroker position update logic."""

    def test_update_position_creates_new_long(self, broker: HybridPaperBroker) -> None:
        """Test creating new long position."""
        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("0.5")
        assert pos.entry_price == Decimal("50000")
        assert pos.side == "long"

    def test_update_position_creates_new_short(self, broker: HybridPaperBroker) -> None:
        """Test creating new short position."""
        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("-0.5")
        assert pos.side == "short"

    def test_update_position_adds_to_long(self, broker: HybridPaperBroker) -> None:
        """Test adding to existing long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("52000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("1.0")
        # Average price: (0.5 * 50000 + 0.5 * 52000) / 1.0 = 51000
        assert pos.entry_price == Decimal("51000")

    def test_update_position_reduces_long(self, broker: HybridPaperBroker) -> None:
        """Test reducing long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("52000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("0.5")
        assert pos.entry_price == Decimal("50000")  # Entry price unchanged on reduction

    def test_update_position_closes_position(self, broker: HybridPaperBroker) -> None:
        """Test closing position completely."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("52000"))

        assert "BTC-USD" not in broker._positions

    def test_short_position_add_recalculates_average(self, broker: HybridPaperBroker) -> None:
        """Test adding to short position recalculates average entry."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("-1"),
            entry_price=Decimal("100"),
            mark_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("1"), Decimal("120"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("-2")
        assert pos.entry_price == Decimal("110")

    def test_short_position_reduce_keeps_entry_price(self, broker: HybridPaperBroker) -> None:
        """Test reducing short position keeps original entry price."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("-1"),
            entry_price=Decimal("100"),
            mark_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("90"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("-0.5")
        assert pos.entry_price == Decimal("100")
