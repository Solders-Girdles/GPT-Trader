"""Tests for StateDeltaUpdater market and positions comparisons."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.state_management.delta_updater import StateDeltaUpdater
from gpt_trader.tui.types import MarketState, PortfolioSummary, Position


class TestStateDeltaUpdaterMarketAndPositions:
    """Test StateDeltaUpdater comparisons for market and positions."""

    def test_compare_market_no_changes(self):
        """Test market comparison with identical data."""
        updater = StateDeltaUpdater()
        market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(market, market)

        assert not result.has_changes

    def test_compare_market_price_change(self):
        """Test market comparison detects price change."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("51000")},
            last_update=1234567891.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.BTC-USD" in result.changed_fields

    def test_compare_market_new_symbol(self):
        """Test market comparison detects new symbol."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.ETH-USD" in result.changed_fields

    def test_compare_market_removed_symbol(self):
        """Test market comparison detects removed symbol."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.ETH-USD" in result.changed_fields

    def test_compare_positions_no_changes(self):
        """Test position comparison with identical data."""
        updater = StateDeltaUpdater()
        positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("1000"),
                    mark_price=Decimal("51000"),
                )
            },
            total_unrealized_pnl=Decimal("1000"),
            equity=Decimal("51000"),
        )

        result = updater.compare_positions(positions, positions)

        assert not result.has_changes

    def test_compare_positions_pnl_change(self):
        """Test position comparison detects P&L change."""
        updater = StateDeltaUpdater()
        old_positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("1000"),
                    mark_price=Decimal("51000"),
                )
            },
            total_unrealized_pnl=Decimal("1000"),
            equity=Decimal("51000"),
        )
        new_positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("2000"),
                    mark_price=Decimal("52000"),
                )
            },
            total_unrealized_pnl=Decimal("2000"),
            equity=Decimal("52000"),
        )

        result = updater.compare_positions(old_positions, new_positions)

        assert result.has_changes
        assert "positions.BTC-USD.unrealized_pnl" in result.changed_fields
