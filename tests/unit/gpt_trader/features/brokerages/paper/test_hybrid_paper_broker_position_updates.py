"""Tests for `HybridPaperBroker` position update logic."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from gpt_trader.core import OrderSide, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerPositionUpdates:
    """Test HybridPaperBroker position update logic."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture."""
        with patch("gpt_trader.features.brokerages.paper.hybrid.CoinbaseClient"):
            with patch("gpt_trader.features.brokerages.paper.hybrid.SimpleAuth"):
                return HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                )

    def test_update_position_creates_new_long(self, broker) -> None:
        """Test creating new long position."""
        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("0.5")
        assert pos.entry_price == Decimal("50000")
        assert pos.side == "long"

    def test_update_position_creates_new_short(self, broker) -> None:
        """Test creating new short position."""
        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("-0.5")
        assert pos.side == "short"

    def test_update_position_adds_to_long(self, broker) -> None:
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

    def test_update_position_reduces_long(self, broker) -> None:
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

    def test_update_position_closes_position(self, broker) -> None:
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
