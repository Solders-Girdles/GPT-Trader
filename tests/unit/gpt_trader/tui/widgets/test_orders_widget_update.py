from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.datatable_test_utils import (  # naming: allow
    create_mock_datatable,  # naming: allow
)
from textual.widgets import DataTable

from gpt_trader.tui.types import Order, Trade
from gpt_trader.tui.widgets.portfolio import OrdersWidget


class TestOrdersWidget:
    def test_update_orders(self) -> None:
        widget = OrdersWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#orders-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )

        orders = [
            Order(
                order_id="ord_123",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
                status="OPEN",
                type="LIMIT",
                time_in_force="GTC",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.025"),
                avg_fill_price=Decimal("49950.00"),
            )
        ]

        widget.update_orders(orders)

        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "50,000.0000"
        assert str(args[4]) == "25%"
        assert "49,950" in str(args[5])
        assert "OPEN" in str(args[7])
        assert kwargs.get("key") == "ord_123"

    def test_update_orders_trade_derived_fill(self) -> None:
        widget = OrdersWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#orders-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )

        orders = [
            Order(
                order_id="ord_456",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("2.0"),
                price=Decimal("3000.00"),
                status="OPEN",
                type="LIMIT",
                time_in_force="GTC",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0"),
                avg_fill_price=None,
            )
        ]

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("3010.00"),
                order_id="ord_456",
                time="2024-01-01T12:00:00Z",
            ),
            Trade(
                trade_id="trd_2",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("3020.00"),
                order_id="ord_456",
                time="2024-01-01T12:01:00Z",
            ),
        ]

        widget.update_orders(orders, trades=trades)

        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert str(args[4]) == "50%"
        assert "3,015" in str(args[5])
        assert kwargs.get("key") == "ord_456"

    def test_trades_stored_for_detail_modal(self) -> None:
        widget = OrdersWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#orders-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )

        orders = [
            Order(
                order_id="ord_123",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="OPEN",
                creation_time=1600000000.0,
            )
        ]

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                order_id="ord_123",
                time="2024-01-01T12:00:00Z",
            )
        ]

        widget.update_orders(orders, trades=trades)

        assert len(widget._trades) == 1
        assert widget._trades[0].trade_id == "trd_1"
