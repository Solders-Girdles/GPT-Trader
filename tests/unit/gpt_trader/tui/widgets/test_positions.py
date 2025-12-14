from decimal import Decimal
from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.events import TradeMatcherResetRequested
from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.types import Order, Position, Trade
from gpt_trader.tui.widgets.portfolio import (
    OrdersWidget,
    PositionsWidget,
    TradesWidget,
)


def create_mock_datatable():
    """Create a properly configured mock DataTable with row-key support."""
    mock_table = MagicMock(spec=DataTable)
    # Mock rows as an empty dict-like object
    mock_table.rows = MagicMock()
    mock_table.rows.keys.return_value = set()  # No existing rows
    mock_table.row_count = 0
    mock_table.columns = MagicMock()
    mock_table.columns.keys.return_value = []
    return mock_table


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

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if (selector == DataTable or "#positions" in str(selector).lower())
                else mock_empty_label
            )
        )

        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
            )
        }

        widget.update_positions(positions, Decimal("100.00"))

        # With row-key optimization, add_row is called with key=symbol
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        # New column order: Symbol, Type, Side, Qty, Entry, Mark, PnL, %, Lev, Liq%
        assert args[0] == "BTC-USD"
        assert "SPOT" in str(args[1])  # Type column
        assert "LONG" in str(args[2])  # Side column
        assert str(args[3]) == "0.5"  # Qty
        assert str(args[4]) == "50,000.0000"  # Entry with commas and 4 decimal places
        assert str(args[5]) == "50,000.0000"  # Mark defaults to entry if mark_price missing
        assert str(args[6]) == "$100.00"  # P&L formatted as currency
        # Leverage check
        assert "1x" in str(args[8])
        # Verify row key is set
        assert kwargs.get("key") == "BTC-USD"


class TestOrdersWidget:
    def test_update_orders(self):
        widget = OrdersWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
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
                creation_time="1600000000",
            )
        ]

        widget.update_orders(orders)

        # With row-key optimization, add_row is called with key=order_id
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "50,000.0000"  # Formatted with commas and 4 decimal places
        assert args[4] == "OPEN"
        # Verify row key is set
        assert kwargs.get("key") == "ord_123"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
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

        # With row-key optimization, add_row is called with key=trade_id
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SELL" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "51,000.0000"  # Formatted with commas and 4 decimal places
        assert args[4] == "ord_123"  # Order ID (short enough to not be truncated)
        # P&L column should be present (args[5]) - unmatched trade shows N/A
        assert "N/A" in str(args[5])
        # Verify row key is set
        assert kwargs.get("key") == "trd_123"

    def test_update_trades_truncation(self):
        widget = TradesWidget()
        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
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
        args, kwargs = mock_table.add_row.call_args
        assert args[4] == long_id[-8:]
        assert kwargs.get("key") == "trd_123"

    def test_update_trades_with_matched_pnl(self):
        """Test that matched BUY/SELL trades show P&L."""
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
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

        # Check that add_row was called twice (for new trades)
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

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
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

        args, kwargs = mock_table.add_row.call_args
        # P&L column (args[5]) should show N/A
        assert "N/A" in str(args[5])
        assert kwargs.get("key") == "trd_1"

    def test_trade_matcher_reset_event_handler(self):
        """Test that TradeMatcherResetRequested event resets the trade matcher."""
        widget = TradesWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        # Match both string selectors and DataTable type
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if ("#trades-table" in str(selector) or selector == DataTable)
                else mock_empty_label
            )
        )
        # Initialize the trade matcher by calling on_mount
        widget.on_mount()

        # Mock the trade matcher's reset method
        widget._trade_matcher.reset = MagicMock()

        # Create and handle the event
        event = TradeMatcherResetRequested()
        widget.on_trade_matcher_reset_requested(event)

        # Verify reset was called
        widget._trade_matcher.reset.assert_called_once()
