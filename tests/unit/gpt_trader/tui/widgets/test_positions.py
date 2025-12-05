from decimal import Decimal
from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.types import Order, Position, Trade
from gpt_trader.tui.widgets.positions import (
    OrdersWidget,
    PositionsWidget,
    TradesWidget,
)


class TestFormatting:
    """Test suite for formatting helper functions (using Decimal types)."""

    def test_format_price(self):
        """Test formatting price with Decimal."""
        result = format_price(Decimal("44520.89963595054"), decimals=4)
        assert result == "44,520.8996"

    def test_format_quantity(self):
        """Test formatting quantity with Decimal."""
        result = format_quantity(Decimal("0.0001234567"), decimals=4)
        # format_quantity strips trailing zeros
        assert "0.0001" in result


class TestPositionsWidget:
    def test_update_positions(self):
        widget = PositionsWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
            )
        }

        widget.update_positions(positions, Decimal("100.00"))

        mock_table.clear.assert_called_once()
        # Verify row content
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        # Quantity, Entry, Current, PnL, %, Leverage are now Text objects
        assert str(args[1]) == "0.5"
        assert str(args[2]) == "50,000.0000"  # Formatted with commas and 4 decimal places
        assert str(args[3]) == "50,000.0000"  # Current defaults to entry if mark_price missing
        assert str(args[4]) == "$100.00"  # P&L formatted as currency with $ sign
        # Leverage check
        assert "1.0x" in str(args[6])


class TestOrdersWidget:
    def test_update_orders(self):
        widget = OrdersWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

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
                creation_time="1600000000",
            )
        ]

        widget.update_orders(orders)

        mock_table.clear.assert_called_once()
        # Verify row content includes Type and TIF
        # Updated to match simplified columns: Symbol, Side, Quantity, Price, Status
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "50,000.0000"  # Formatted with commas and 4 decimal places
        assert args[4] == "OPEN"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)
        # Initialize the trade matcher by calling on_mount
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

        mock_table.clear.assert_called_once()
        # Verify row content includes Order ID and P&L (7 columns total now)
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SELL" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "51,000.0000"  # Formatted with commas and 4 decimal places
        assert args[4] == "ord_123"  # Order ID (short enough to not be truncated)
        # P&L column should be present (args[5]) - unmatched trade shows N/A
        assert "N/A" in str(args[5])
        # Time is args[6]

    def test_update_trades_truncation(self):
        widget = TradesWidget()
        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)
        # Initialize the trade matcher by calling on_mount
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
        args, _ = mock_table.add_row.call_args
        assert args[4] == long_id[-8:]

    def test_update_trades_with_matched_pnl(self):
        """Test that matched BUY/SELL trades show P&L."""
        widget = TradesWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)
        # Initialize the trade matcher by calling on_mount
        widget.on_mount()

        # Create a matched pair: BUY then SELL
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

        # Check that add_row was called twice
        assert mock_table.add_row.call_count == 2

        # First call (BUY trade) should show P&L
        first_call_args = mock_table.add_row.call_args_list[0][0]
        assert "BTC-USD" in first_call_args[0]
        pnl_text = str(first_call_args[5])
        # Should show positive P&L (profit)
        assert "+" in pnl_text or "939" in pnl_text

        # Second call (SELL trade) should also show P&L
        second_call_args = mock_table.add_row.call_args_list[1][0]
        pnl_text = str(second_call_args[5])
        assert "+" in pnl_text or "939" in pnl_text

    def test_update_trades_unmatched_shows_na(self):
        """Test that unmatched trades show N/A for P&L."""
        widget = TradesWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)
        # Initialize the trade matcher by calling on_mount
        widget.on_mount()

        # Single BUY trade with no matching SELL
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

        args, _ = mock_table.add_row.call_args
        # P&L column (args[5]) should show N/A
        assert "N/A" in str(args[5])
