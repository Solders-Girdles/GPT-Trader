"""Tests for StateDeltaUpdater order and trade comparisons."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.state_management.delta_updater import StateDeltaUpdater
from gpt_trader.tui.types import ActiveOrders, Order, Trade, TradeHistory


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

    def test_compare_orders_status_change(self):
        """Test order comparison detects status change."""
        updater = StateDeltaUpdater()
        old_orders = ActiveOrders(
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
        new_orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="FILLED",
                )
            ]
        )

        result = updater.compare_orders(old_orders, new_orders)

        assert result.has_changes
        assert "orders.order-1.status" in result.changed_fields

    def test_compare_orders_new_order(self):
        """Test order comparison detects new order."""
        updater = StateDeltaUpdater()
        old_orders = ActiveOrders(orders=[])
        new_orders = ActiveOrders(
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

        result = updater.compare_orders(old_orders, new_orders)

        assert result.has_changes
        assert "orders.order-1" in result.changed_fields

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
