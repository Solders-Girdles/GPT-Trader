"""Tests for StateDeltaUpdater market, position, order, and trade comparisons."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.tui.state_management.delta_updater import StateDeltaUpdater
from gpt_trader.tui.types import (
    ActiveOrders,
    MarketState,
    Order,
    PortfolioSummary,
    Position,
    Trade,
    TradeHistory,
)


def _build_order(status: str) -> Order:
    return Order(
        order_id="order-1",
        symbol="BTC-USD",
        side="BUY",
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        status=status,
    )


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

    @pytest.mark.parametrize(
        ("old_prices", "new_prices", "changed_field", "new_update"),
        [
            (
                {"BTC-USD": Decimal("50000")},
                {"BTC-USD": Decimal("51000")},
                "prices.BTC-USD",
                1234567891.0,
            ),
            (
                {"BTC-USD": Decimal("50000")},
                {"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
                "prices.ETH-USD",
                1234567890.0,
            ),
            (
                {"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
                {"BTC-USD": Decimal("50000")},
                "prices.ETH-USD",
                1234567890.0,
            ),
        ],
    )
    def test_compare_market_changes(self, old_prices, new_prices, changed_field, new_update):
        """Test market comparison detects price or symbol changes."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices=old_prices,
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices=new_prices,
            last_update=new_update,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert changed_field in result.changed_fields

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


class TestStateDeltaUpdaterOrdersAndTrades:
    """Test StateDeltaUpdater comparisons for orders and trades."""

    def test_compare_orders_no_changes(self):
        """Test order comparison with identical data."""
        updater = StateDeltaUpdater()
        orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="OPEN",
                )
            ]
        )

        result = updater.compare_orders(orders, orders)

        assert not result.has_changes

    @pytest.mark.parametrize(
        ("old_orders", "new_orders", "changed_field"),
        [
            (
                ActiveOrders(orders=[_build_order("OPEN")]),
                ActiveOrders(orders=[_build_order("FILLED")]),
                "orders.order-1.status",
            ),
            (
                ActiveOrders(orders=[]),
                ActiveOrders(orders=[_build_order("OPEN")]),
                "orders.order-1",
            ),
        ],
    )
    def test_compare_orders_changes(self, old_orders, new_orders, changed_field):
        """Test order comparison detects order changes."""
        updater = StateDeltaUpdater()

        result = updater.compare_orders(old_orders, new_orders)

        assert result.has_changes
        assert changed_field in result.changed_fields

    def test_compare_trades_no_changes(self):
        """Test trade comparison with identical data."""
        updater = StateDeltaUpdater()
        trades = TradeHistory(
            trades=[
                Trade(
                    trade_id="trade-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    order_id="order-1",
                    time="2024-01-01T00:00:00Z",
                )
            ]
        )

        result = updater.compare_trades(trades, trades)

        assert not result.has_changes

    def test_compare_trades_new_trade(self):
        """Test trade comparison detects new trade."""
        updater = StateDeltaUpdater()
        old_trades = TradeHistory(trades=[])
        new_trades = TradeHistory(
            trades=[
                Trade(
                    trade_id="trade-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    order_id="order-1",
                    time="2024-01-01T00:00:00Z",
                )
            ]
        )

        result = updater.compare_trades(old_trades, new_trades)

        assert result.has_changes
        assert "trades.trade-1" in result.changed_fields
