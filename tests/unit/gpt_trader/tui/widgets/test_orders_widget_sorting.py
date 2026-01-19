from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.datatable_test_utils import (  # naming: allow
    create_mock_datatable,  # naming: allow
)
from textual.widgets import DataTable

from gpt_trader.tui.types import Order
from gpt_trader.tui.widgets.portfolio import OrdersWidget


class TestOrdersWidgetSorting:
    def test_sort_cycle(self) -> None:
        widget = OrdersWidget()
        widget.notify = MagicMock()

        assert widget.sort_column is None
        assert widget.sort_ascending is True

        widget.action_cycle_sort()
        assert widget.sort_column == "fill_pct"
        assert widget.sort_ascending is True

        widget.action_cycle_sort()
        assert widget.sort_column == "fill_pct"
        assert widget.sort_ascending is False

        widget.action_cycle_sort()
        assert widget.sort_column == "age"
        assert widget.sort_ascending is True

        widget.action_cycle_sort()
        assert widget.sort_column == "age"
        assert widget.sort_ascending is False

        widget.action_cycle_sort()
        assert widget.sort_column is None
        assert widget.sort_ascending is True

    def test_sort_orders_by_fill_pct(self) -> None:
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
                order_id="ord_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.25"),
                avg_fill_price=Decimal("50000.00"),
            ),
            Order(
                order_id="ord_2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("3000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.75"),
                avg_fill_price=Decimal("3000.00"),
            ),
            Order(
                order_id="ord_3",
                symbol="SOL-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("100.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.50"),
                avg_fill_price=Decimal("100.00"),
            ),
        ]

        widget.sort_column = "fill_pct"
        widget.sort_ascending = True
        widget.update_orders(orders)

        call_keys = [call[1]["key"] for call in mock_table.add_row.call_args_list]
        assert call_keys == ["ord_1", "ord_3", "ord_2"]

    def test_sort_orders_by_fill_pct_descending(self) -> None:
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
                order_id="ord_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.25"),
                avg_fill_price=Decimal("50000.00"),
            ),
            Order(
                order_id="ord_2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("3000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.75"),
                avg_fill_price=Decimal("3000.00"),
            ),
            Order(
                order_id="ord_3",
                symbol="SOL-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("100.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.50"),
                avg_fill_price=Decimal("100.00"),
            ),
        ]

        widget.sort_column = "fill_pct"
        widget.sort_ascending = False
        widget.update_orders(orders)

        call_keys = [call[1]["key"] for call in mock_table.add_row.call_args_list]
        assert call_keys == ["ord_2", "ord_3", "ord_1"]

    def test_action_show_order_detail_no_orders(self) -> None:
        widget = OrdersWidget()
        widget.notify = MagicMock()

        mock_table = create_mock_datatable()
        mock_table.row_count = 0
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#orders-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )

        widget.action_show_order_detail()

        widget.notify.assert_called_once()
        assert "No orders" in widget.notify.call_args[0][0]
