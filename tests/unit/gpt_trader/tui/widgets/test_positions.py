# naming: allow - qty is standard trading abbreviation for quantity

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
from gpt_trader.tui.widgets.portfolio.order_detail_modal import OrderDetailModal


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
                creation_time=1600000000.0,  # Float timestamp for age calculation
                filled_quantity=Decimal("0.025"),  # 25% filled
                avg_fill_price=Decimal("49950.00"),  # Avg fill price
            )
        ]

        widget.update_orders(orders)

        # With row-key optimization, add_row is called with key=order_id
        # Columns: Symbol, Side, Qty, Price, Fill%, Avg Px, Age, Status
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert str(args[2]) == "0.1"  # Qty
        assert str(args[3]) == "50,000.0000"  # Price
        assert str(args[4]) == "25%"  # Fill%
        assert "49,950" in str(args[5])  # Avg Px
        # args[6] is Age (Rich Text)
        # args[7] is Status (Rich Text with color)
        assert "OPEN" in str(args[7])
        # Verify row key is set
        assert kwargs.get("key") == "ord_123"

    def test_update_orders_trade_derived_fill(self):
        """Test that fill info can be derived from trades when order fields are empty."""
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

        # Order without fill info
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
                filled_quantity=Decimal("0"),  # No fill info on order
                avg_fill_price=None,
            )
        ]

        # Trades that link to the order
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
        # Fill% should be 50% (1.0 filled out of 2.0)
        assert str(args[4]) == "50%"
        # Avg Px should be weighted average: (0.5*3010 + 0.5*3020) / 1.0 = 3015
        assert "3,015" in str(args[5])
        assert kwargs.get("key") == "ord_456"

    def test_sort_cycle(self):
        """Test that sort cycling works: None → Fill% ↑ → Fill% ↓ → Age ↑ → Age ↓ → None."""
        widget = OrdersWidget()
        # Mock notify to avoid NoActiveAppError
        widget.notify = MagicMock()

        # Initial state: no sorting
        assert widget.sort_column is None
        assert widget.sort_ascending is True

        # First cycle: Fill% ascending
        widget.action_cycle_sort()
        assert widget.sort_column == "fill_pct"
        assert widget.sort_ascending is True

        # Second cycle: Fill% descending
        widget.action_cycle_sort()
        assert widget.sort_column == "fill_pct"
        assert widget.sort_ascending is False

        # Third cycle: Age ascending
        widget.action_cycle_sort()
        assert widget.sort_column == "age"
        assert widget.sort_ascending is True

        # Fourth cycle: Age descending
        widget.action_cycle_sort()
        assert widget.sort_column == "age"
        assert widget.sort_ascending is False

        # Fifth cycle: back to None
        widget.action_cycle_sort()
        assert widget.sort_column is None
        assert widget.sort_ascending is True

    def test_sort_orders_by_fill_pct(self):
        """Test that orders can be sorted by fill percentage."""
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

        # Create orders with different fill percentages
        orders = [
            Order(
                order_id="ord_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.25"),  # 25% filled
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
                filled_quantity=Decimal("0.75"),  # 75% filled
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
                filled_quantity=Decimal("0.50"),  # 50% filled
                avg_fill_price=Decimal("100.00"),
            ),
        ]

        # Set sort to Fill% ascending (lowest first) before loading data
        widget.sort_column = "fill_pct"
        widget.sort_ascending = True
        widget.update_orders(orders)

        # Get the order of add_row calls
        call_keys = [call[1]["key"] for call in mock_table.add_row.call_args_list]
        # Should be: ord_1 (25%), ord_3 (50%), ord_2 (75%)
        assert call_keys == ["ord_1", "ord_3", "ord_2"]

    def test_sort_orders_by_fill_pct_descending(self):
        """Test that orders can be sorted by fill percentage descending."""
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

        # Create orders with different fill percentages
        orders = [
            Order(
                order_id="ord_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="OPEN",
                creation_time=1600000000.0,
                filled_quantity=Decimal("0.25"),  # 25% filled
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
                filled_quantity=Decimal("0.75"),  # 75% filled
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
                filled_quantity=Decimal("0.50"),  # 50% filled
                avg_fill_price=Decimal("100.00"),
            ),
        ]

        # Set sort to Fill% descending (highest first)
        widget.sort_column = "fill_pct"
        widget.sort_ascending = False
        widget.update_orders(orders)

        call_keys = [call[1]["key"] for call in mock_table.add_row.call_args_list]
        # Should be: ord_2 (75%), ord_3 (50%), ord_1 (25%)
        assert call_keys == ["ord_2", "ord_3", "ord_1"]

    def test_action_show_order_detail_no_orders(self):
        """Test that action_show_order_detail notifies when no orders."""
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

    def test_trades_stored_for_detail_modal(self):
        """Test that trades are stored for access by detail modal."""
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

        # Verify trades are stored
        assert len(widget._trades) == 1
        assert widget._trades[0].trade_id == "trd_1"


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
        # Column order: Symbol, Links, Side, Quantity, Price, Order ID, P&L, Time
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        # args[1] is Links column (badge text)
        assert "SELL" in str(args[2])  # Side at index 2
        assert str(args[3]) == "0.1"  # Quantity at index 3
        assert str(args[4]) == "51,000.0000"  # Price at index 4
        assert args[5] == "ord_123"  # Order ID at index 5
        # P&L column should be present (args[6]) - unmatched trade shows N/A
        assert "N/A" in str(args[6])
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
        # Column order: Symbol, Links, Side, Quantity, Price, Order ID, P&L, Time
        assert args[5] == long_id[-8:]  # Order ID at index 5
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

        # Column order: Symbol, Links, Side, Quantity, Price, Order ID, P&L, Time
        # First call (BUY trade) should show P&L
        first_call_args = mock_table.add_row.call_args_list[0][0]
        assert "BTC-USD" in first_call_args[0]
        pnl_text = str(first_call_args[6])  # P&L at index 6
        # Should show positive P&L (profit)
        assert "+" in pnl_text or "939" in pnl_text

        # Second call (SELL trade) should also show P&L
        second_call_args = mock_table.add_row.call_args_list[1][0]
        pnl_text = str(second_call_args[6])  # P&L at index 6
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
        # Column order: Symbol, Links, Side, Quantity, Price, Order ID, P&L, Time
        # P&L column (args[6]) should show N/A
        assert "N/A" in str(args[6])
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

    def test_linkage_badges_with_no_state(self):
        """Test that linkage badges show '--' when no state is provided."""
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

        # Update without state
        widget.update_trades(trades)

        args, kwargs = mock_table.add_row.call_args
        # Links column (index 1) should show dim badge when no state
        links_text = str(args[1])
        assert "⬚" in links_text  # Order badge present

    def test_symbol_filter_filters_trades(self):
        """Test that symbol filter correctly filters trades."""
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

        # No filter - both trades shown
        widget.update_trades(trades)
        assert mock_table.add_row.call_count == 2

        mock_table.reset_mock()

        # Apply BTC filter
        widget.symbol_filter = "BTC-USD"
        widget.update_trades(trades)
        assert mock_table.add_row.call_count == 1

    def test_symbol_filter_cycle(self):
        """Test cycling through symbol filters."""
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

        # Initially no filter
        assert widget.symbol_filter == ""

        # Cycle to first symbol
        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == "BTC-USD"

        # Cycle to second symbol
        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == "ETH-USD"

        # Cycle back to no filter
        widget.action_cycle_symbol_filter()
        assert widget.symbol_filter == ""

    def test_clear_filters(self):
        """Test clearing all filters."""
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

        # Clear filters
        widget.action_clear_filters()
        assert widget.symbol_filter == ""


class TestOrderDetailModal:
    """Tests for OrderDetailModal."""

    def test_init_with_order(self):
        """Test modal initialization with order."""
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            type="LIMIT",
            time_in_force="GTC",
            creation_time=1600000000.0,
            filled_quantity=Decimal("0.5"),
            avg_fill_price=Decimal("49900.00"),
        )

        modal = OrderDetailModal(order)

        assert modal.order == order
        assert modal.trades == []
        assert modal._linked_trades == []

    def test_init_with_trades(self):
        """Test modal initialization with order and trades."""
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                order_id="ord_123",
                time="2024-01-01T12:00:00Z",
                fee=Decimal("25.00"),
            ),
            Trade(
                trade_id="trd_2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50100.00"),
                order_id="ord_456",  # Different order
                time="2024-01-01T12:01:00Z",
                fee=Decimal("25.05"),
            ),
        ]

        modal = OrderDetailModal(order, trades=trades)

        assert modal.order == order
        assert len(modal.trades) == 2

    def test_calculate_fill_pct(self):
        """Test fill percentage calculation."""
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("2.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
            filled_quantity=Decimal("0.5"),  # 25% filled
        )

        modal = OrderDetailModal(order)
        result = modal._calculate_fill_pct()

        assert result == "25%"

    def test_calculate_total_fees(self):
        """Test total fees calculation from linked trades."""
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                order_id="ord_123",
                time="2024-01-01T12:00:00Z",
                fee=Decimal("25.00"),
            ),
            Trade(
                trade_id="trd_2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50100.00"),
                order_id="ord_123",
                time="2024-01-01T12:01:00Z",
                fee=Decimal("25.05"),
            ),
        ]

        modal = OrderDetailModal(order, trades=trades)
        # Manually set _linked_trades since compose() isn't called
        modal._linked_trades = [t for t in trades if t.order_id == order.order_id]

        result = modal._calculate_total_fees()

        assert result == Decimal("50.05")

    def test_format_trade_time(self):
        """Test trade time formatting."""
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        modal = OrderDetailModal(order)

        # Full ISO format
        assert modal._format_trade_time("2024-01-15T14:30:45.123Z") == "14:30:45"

        # Without milliseconds
        assert modal._format_trade_time("2024-01-15T09:05:00Z") == "09:05:00"


class TestPositionDetailModal:
    """Tests for PositionDetailModal."""

    def test_init_with_position(self):
        """Test modal initialization with position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
            side="long",
        )

        modal = PositionDetailModal(position)

        assert modal.position == position
        assert modal.trades == []
        assert modal._linked_trades == []

    def test_init_with_trades(self):
        """Test modal initialization with trades list."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
                fee=Decimal("10.00"),
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",  # Different symbol
                side="BUY",
                quantity=Decimal("5.0"),
                price=Decimal("2500.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
                fee=Decimal("5.00"),
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)

        assert modal.position == position
        assert len(modal.trades) == 2

    def test_calculate_total_fees(self):
        """Test fee calculation from linked trades."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
                fee=Decimal("10.50"),
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
                fee=Decimal("15.25"),
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        # Manually set _linked_trades since compose() isn't called
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_total_fees()

        assert result == Decimal("25.75")

    def test_calculate_realized_pnl_long_position(self):
        """Test realized P&L calculation for long position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),  # Remaining after partial close
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("1000.00"),
            side="long",
        )

        trades = [
            Trade(  # Opening buy
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(  # Closing sell - profit
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("55000.00"),  # Sold higher than entry
                order_id="o2",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_realized_pnl()

        # Realized = (55000 - 50000) * 0.5 = 2500
        assert result == Decimal("2500.00")

    def test_calculate_realized_pnl_short_position(self):
        """Test realized P&L calculation for short position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("2800.00"),
            unrealized_pnl=Decimal("400.00"),
            side="short",
        )

        trades = [
            Trade(  # Opening short (sell)
                trade_id="t1",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("3.0"),
                price=Decimal("3000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(  # Closing partial (buy to cover) - profit
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("2700.00"),  # Bought lower than entry
                order_id="o2",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_realized_pnl()

        # Realized for short = (entry - buy_price) * qty = (3000 - 2700) * 1.0 = 300
        assert result == Decimal("300.00")

    def test_categorize_trades_long(self):
        """Test trade categorization for long position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.5"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("51000.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = trades

        opens, closes = modal._categorize_trades()

        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].side == "BUY"
        assert closes[0].side == "SELL"

    def test_categorize_trades_short(self):
        """Test trade categorization for short position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("3000.00"),
            unrealized_pnl=Decimal("0"),
            side="short",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="ETH-USD",
                side="SELL",  # Opens short
                quantity=Decimal("2.0"),
                price=Decimal("3000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",  # Closes short
                quantity=Decimal("1.0"),
                price=Decimal("2900.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = trades

        opens, closes = modal._categorize_trades()

        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].side == "SELL"
        assert closes[0].side == "BUY"

    def test_get_trade_type_long_position(self):
        """Test trade type determination for long position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        buy_trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )

        sell_trade = Trade(
            trade_id="t2",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("51000.00"),
            order_id="o2",
            time="2024-01-15T11:00:00Z",
        )

        modal = PositionDetailModal(position)

        assert modal._get_trade_type(buy_trade) == "OPEN"
        assert modal._get_trade_type(sell_trade) == "CLOSE"

    def test_get_trade_type_short_position(self):
        """Test trade type determination for short position."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("3000.00"),
            unrealized_pnl=Decimal("0"),
            side="short",
        )

        sell_trade = Trade(
            trade_id="t1",
            symbol="ETH-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("3000.00"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )

        buy_trade = Trade(
            trade_id="t2",
            symbol="ETH-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("2900.00"),
            order_id="o2",
            time="2024-01-15T11:00:00Z",
        )

        modal = PositionDetailModal(position)

        assert modal._get_trade_type(sell_trade) == "OPEN"
        assert modal._get_trade_type(buy_trade) == "CLOSE"

    def test_format_trade_time(self):
        """Test trade time formatting."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        modal = PositionDetailModal(position)

        # Full ISO format
        assert modal._format_trade_time("2024-01-15T14:30:45.123Z") == "14:30:45"

        # Without milliseconds
        assert modal._format_trade_time("2024-01-15T09:05:00Z") == "09:05:00"

    def test_filters_trades_by_symbol(self):
        """Test that only trades matching position symbol are linked."""
        from gpt_trader.tui.widgets.portfolio.position_detail_modal import (
            PositionDetailModal,
        )

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",  # Different symbol
                side="BUY",
                quantity=Decimal("5.0"),
                price=Decimal("2500.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("51000.00"),
                order_id="o3",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        # Simulate what compose() does
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        # Should only have BTC-USD trades
        assert len(modal._linked_trades) == 2
        assert all(t.symbol == "BTC-USD" for t in modal._linked_trades)
