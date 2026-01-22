"""Tests for HybridPaperBroker position update logic."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import OrderSide, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerPositionUpdates:
    """Test HybridPaperBroker position update logic."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture."""
        return broker_factory()

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
