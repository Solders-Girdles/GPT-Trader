from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.datatable_test_utils import (  # naming: allow
    create_mock_datatable,  # naming: allow
)
from textual.widgets import DataTable

from gpt_trader.tui.types import Trade
from gpt_trader.tui.widgets.portfolio import TradesWidget


class TestTradesWidget:
    def test_update_trades(self) -> None:
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
                trade_id="trd_123",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000.00"),
                order_id="ord_123",
                time="2023-01-01T12:00:00.000Z",
            )
        ]

        widget.update_trades(trades)

        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SELL" in str(args[2])
        assert str(args[3]) == "0.1"
        assert str(args[4]) == "51,000.0000"
        assert args[5] == "ord_123"
        assert "N/A" in str(args[6])
        assert kwargs.get("key") == "trd_123"

    def test_update_trades_truncation(self) -> None:
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

        long_id = "12345678-1234-1234-1234-1234567890ab"
        trades = [
            Trade(
                trade_id="trd_123",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000.00"),
                order_id=long_id,
                time="2023-01-01T12:00:00.000Z",
            )
        ]

        widget.update_trades(trades)

        args, kwargs = mock_table.add_row.call_args
        assert args[5] == long_id[-8:]
        assert kwargs.get("key") == "trd_123"

    def test_update_trades_with_matched_pnl(self) -> None:
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
                fee=Decimal("30.00"),
            ),
            Trade(
                trade_id="trd_2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("1.0"),
                price=Decimal("51000.00"),
                order_id="ord_2",
                time="2023-01-01T12:01:00.000Z",
                fee=Decimal("30.60"),
            ),
        ]

        widget.update_trades(trades)

        assert mock_table.add_row.call_count == 2
        first_call_args = mock_table.add_row.call_args_list[0][0]
        assert "BTC-USD" in first_call_args[0]
        assert "+" in str(first_call_args[6]) or "939" in str(first_call_args[6])

        second_call_args = mock_table.add_row.call_args_list[1][0]
        assert "+" in str(second_call_args[6]) or "939" in str(second_call_args[6])

    def test_update_trades_unmatched_shows_na(self) -> None:
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
                fee=Decimal("30.00"),
            ),
        ]

        widget.update_trades(trades)

        args, kwargs = mock_table.add_row.call_args
        assert "N/A" in str(args[6])
        assert kwargs.get("key") == "trd_1"
