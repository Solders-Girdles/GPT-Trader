from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.datatable_test_utils import (  # naming: allow
    create_mock_datatable,  # naming: allow
)
from textual.widgets import DataTable

from gpt_trader.tui.events import TradeMatcherResetRequested
from gpt_trader.tui.types import Trade
from gpt_trader.tui.widgets.portfolio import TradesWidget


class TestTradesWidgetFilters:
    def test_trade_matcher_reset_event_handler(self) -> None:
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        widget.on_mount()

        widget._trade_matcher.reset = MagicMock()

        event = TradeMatcherResetRequested()
        widget.on_trade_matcher_reset_requested(event)

        widget._trade_matcher.reset.assert_called_once()

    def test_linkage_badges_with_no_state(self) -> None:
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        widget.on_mount()

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="ord_1",
                time="2023-01-01T12:00:00.000Z",
            ),
        ]

        widget.update_trades(trades)

        args, kwargs = mock_table.add_row.call_args
        links_text = str(args[1])
        assert "⬚" in links_text
        assert kwargs.get("key") == "trd_1"

    def test_symbol_filter_filters_trades(self) -> None:
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        widget.on_mount()

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="ord_1",
                time="2023-01-01T12:00:00.000Z",
            ),
            Trade(
                trade_id="trd_2",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("2.0"),
                price=Decimal("2500.00"),
                order_id="ord_2",
                time="2023-01-01T12:01:00.000Z",
            ),
        ]

        widget.update_trades(trades)
        assert mock_table.add_row.call_count == 2

        mock_table.reset_mock()

        widget.symbol_filter = "BTC-USD"
        widget.update_trades(trades)
        assert mock_table.add_row.call_count == 1

    def test_symbol_filter_cycle(self) -> None:
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        widget.on_mount()

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="ord_1",
                time="2023-01-01T12:00:00.000Z",
            ),
            Trade(
                trade_id="trd_2",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("2.0"),
                price=Decimal("2500.00"),
                order_id="ord_2",
                time="2023-01-01T12:01:00.000Z",
            ),
        ]

        widget.update_trades(trades)

        assert widget.symbol_filter == ""

        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == "BTC-USD"

        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == "ETH-USD"

        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == ""

    def test_clear_filters(self) -> None:
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        widget.on_mount()

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="ord_1",
                time="2023-01-01T12:00:00.000Z",
            ),
        ]

        widget.update_trades(trades)
        widget.symbol_filter = "BTC-USD"

        widget.action_clear_filters()
        assert widget.symbol_filter == ""
