"""
Comprehensive tests for PnLRestMixin.

Covers fill processing, position PnL calculation, and portfolio aggregation.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.rest.pnl import PnLRestMixin
from bot_v2.features.brokerages.coinbase.utilities import PositionState


class PnLRestMixinImpl(PnLRestMixin):
    """Implementation with required attributes for testing."""

    def __init__(self):
        self._positions = {}
        self._update_position_metrics = Mock()
        self.market_data = Mock()
        self._event_store = Mock()


@pytest.fixture
def pnl_mixin():
    """Create PnL mixin instance."""
    return PnLRestMixinImpl()


class TestProcessFillForPnL:
    """Test process_fill_for_pnl method."""

    def test_process_fill_opens_long_position(self, pnl_mixin):
        """Should open new long position on buy fill."""
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.5",
            "price": "50000",
            "side": "buy",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        assert "BTC-USD-PERP" in pnl_mixin._positions
        position = pnl_mixin._positions["BTC-USD-PERP"]
        assert position.side == "long"
        assert position.quantity == Decimal("0.5")
        assert position.entry_price == Decimal("50000")

    def test_process_fill_opens_short_position(self, pnl_mixin):
        """Should open new short position on sell fill."""
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.3",
            "price": "50000",
            "side": "sell",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        position = pnl_mixin._positions["BTC-USD-PERP"]
        assert position.side == "short"
        assert position.quantity == Decimal("0.3")

    def test_process_fill_adds_to_position(self, pnl_mixin):
        """Should add to existing position."""
        # Open position
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
        )

        # Add to position
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.3",
            "price": "51000",
            "side": "buy",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        position = pnl_mixin._positions["BTC-USD-PERP"]
        assert position.quantity == Decimal("0.8")

    def test_process_fill_reduces_position(self, pnl_mixin):
        """Should reduce position and realize PnL."""
        # Open long position
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        # Close half at profit
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.5",
            "price": "51000",
            "side": "sell",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        position = pnl_mixin._positions["BTC-USD-PERP"]
        assert position.quantity == Decimal("0.5")
        # Should have realized profit
        assert position.realized_pnl > 0

    def test_process_fill_missing_product_id(self, pnl_mixin):
        """Should ignore fill without product_id."""
        fill = {"size": "0.5", "price": "50000", "side": "buy"}

        pnl_mixin.process_fill_for_pnl(fill)

        assert len(pnl_mixin._positions) == 0

    def test_process_fill_zero_quantity(self, pnl_mixin):
        """Should ignore fill with zero quantity."""
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0",
            "price": "50000",
            "side": "buy",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        assert len(pnl_mixin._positions) == 0

    def test_process_fill_zero_price(self, pnl_mixin):
        """Should ignore fill with zero price."""
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.5",
            "price": "0",
            "side": "buy",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        assert len(pnl_mixin._positions) == 0

    def test_process_fill_updates_metrics(self, pnl_mixin):
        """Should call _update_position_metrics."""
        fill = {
            "product_id": "BTC-USD-PERP",
            "size": "0.5",
            "price": "50000",
            "side": "buy",
        }

        pnl_mixin.process_fill_for_pnl(fill)

        pnl_mixin._update_position_metrics.assert_called_once_with("BTC-USD-PERP")


class TestGetPositionPnL:
    """Test get_position_pnl method."""

    def test_get_position_pnl_no_position(self, pnl_mixin):
        """Should return zero PnL for non-existent position."""
        pnl = pnl_mixin.get_position_pnl("BTC-USD-PERP")

        assert pnl["symbol"] == "BTC-USD-PERP"
        assert pnl["quantity"] == Decimal("0")
        assert pnl["side"] is None
        assert pnl["entry"] is None
        assert pnl["unrealized_pnl"] == Decimal("0")
        assert pnl["realized_pnl"] == Decimal("0")

    def test_get_position_pnl_with_position(self, pnl_mixin):
        """Should calculate PnL for existing position."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        # Set mark price
        pnl_mixin.market_data.get_mark.return_value = Decimal("51000")
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_position_pnl("BTC-USD-PERP")

        assert pnl["quantity"] == Decimal("1.0")
        assert pnl["side"] == "long"
        assert pnl["entry"] == Decimal("50000")
        assert pnl["mark"] == Decimal("51000")
        # Unrealized: (51000 - 50000) * 1.0 = 1000
        assert pnl["unrealized_pnl"] == Decimal("1000")

    def test_get_position_pnl_with_funding(self, pnl_mixin):
        """Should include funding accruals."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = Decimal("50000")

        # Mock funding events
        funding_events = [
            {"type": "funding", "symbol": "BTC-USD-PERP", "funding_amount": "-5.50"},
            {"type": "funding", "symbol": "BTC-USD-PERP", "funding_amount": "-3.25"},
        ]
        pnl_mixin._event_store.tail.return_value = funding_events

        pnl = pnl_mixin.get_position_pnl("BTC-USD-PERP")

        # Total funding: -5.50 + -3.25 = -8.75
        assert pnl["funding_accrued"] == Decimal("-8.75")

    def test_get_position_pnl_no_mark_price(self, pnl_mixin):
        """Should handle missing mark price."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = None
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_position_pnl("BTC-USD-PERP")

        # Should use 0 for mark
        assert pnl["mark"] == Decimal("0")

    def test_get_position_pnl_filters_funding_events(self, pnl_mixin):
        """Should filter funding events by symbol."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = Decimal("50000")

        # Mix of events for different symbols
        events = [
            {"type": "funding", "symbol": "BTC-USD-PERP", "funding_amount": "-5.00"},
            {"type": "funding", "symbol": "ETH-USD-PERP", "funding_amount": "-10.00"},
            {"type": "trade", "symbol": "BTC-USD-PERP"},
        ]
        pnl_mixin._event_store.tail.return_value = events

        pnl = pnl_mixin.get_position_pnl("BTC-USD-PERP")

        # Should only count BTC funding
        assert pnl["funding_accrued"] == Decimal("-5.00")


class TestGetPortfolioPnL:
    """Test get_portfolio_pnl method."""

    def test_get_portfolio_pnl_empty(self, pnl_mixin):
        """Should return zero PnL for empty portfolio."""
        pnl = pnl_mixin.get_portfolio_pnl()

        assert pnl["total_unrealized_pnl"] == Decimal("0")
        assert pnl["total_realized_pnl"] == Decimal("0")
        assert pnl["total_funding"] == Decimal("0")
        assert pnl["total_pnl"] == Decimal("0")
        assert pnl["positions"] == {}

    def test_get_portfolio_pnl_single_position(self, pnl_mixin):
        """Should calculate PnL for single position."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = Decimal("51000")
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_portfolio_pnl()

        assert pnl["total_unrealized_pnl"] == Decimal("1000")
        assert "BTC-USD-PERP" in pnl["positions"]

    def test_get_portfolio_pnl_multiple_positions(self, pnl_mixin):
        """Should aggregate PnL across multiple positions."""
        # Long BTC
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        # Short ETH
        pnl_mixin._positions["ETH-USD-PERP"] = PositionState(
            symbol="ETH-USD-PERP",
            side="short",
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000"),
        )

        # Setup mark prices
        def get_mark(symbol):
            if symbol == "BTC-USD-PERP":
                return Decimal("51000")  # +1000 profit
            if symbol == "ETH-USD-PERP":
                return Decimal("2900")  # +1000 profit (short)
            return Decimal("0")

        pnl_mixin.market_data.get_mark.side_effect = get_mark
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_portfolio_pnl()

        # Total unrealized: 1000 (BTC) + 1000 (ETH) = 2000
        assert pnl["total_unrealized_pnl"] == Decimal("2000")
        assert len(pnl["positions"]) == 2

    def test_get_portfolio_pnl_includes_realized(self, pnl_mixin):
        """Should include realized PnL in total."""
        position = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )
        position.realized_pnl = Decimal("500")  # Set realized PnL
        pnl_mixin._positions["BTC-USD-PERP"] = position

        pnl_mixin.market_data.get_mark.return_value = Decimal("51000")
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_portfolio_pnl()

        assert pnl["total_realized_pnl"] == Decimal("500")
        assert pnl["total_unrealized_pnl"] == Decimal("1000")
        # Total = realized + unrealized
        assert pnl["total_pnl"] == Decimal("1500")

    def test_get_portfolio_pnl_includes_funding(self, pnl_mixin):
        """Should include funding in portfolio total."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = Decimal("50000")

        funding_events = [
            {"type": "funding", "symbol": "BTC-USD-PERP", "funding_amount": "-10.50"},
        ]
        pnl_mixin._event_store.tail.return_value = funding_events

        pnl = pnl_mixin.get_portfolio_pnl()

        assert pnl["total_funding"] == Decimal("-10.50")

    def test_get_portfolio_pnl_breakdown(self, pnl_mixin):
        """Should provide per-position breakdown."""
        pnl_mixin._positions["BTC-USD-PERP"] = PositionState(
            symbol="BTC-USD-PERP",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        pnl_mixin.market_data.get_mark.return_value = Decimal("51000")
        pnl_mixin._event_store.tail.return_value = []

        pnl = pnl_mixin.get_portfolio_pnl()

        assert "BTC-USD-PERP" in pnl["positions"]
        btc_pnl = pnl["positions"]["BTC-USD-PERP"]
        assert btc_pnl["quantity"] == Decimal("1.0")
        assert btc_pnl["unrealized_pnl"] == Decimal("1000")
