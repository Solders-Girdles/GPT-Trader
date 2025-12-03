from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import Order, Position, Trade
from gpt_trader.tui.widgets.positions import OrdersWidget, PositionsWidget, TradesWidget


class TestPositionsWidget:
    def test_update_positions(self):
        widget = PositionsWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD", quantity="0.5", entry_price="50000.00", unrealized_pnl="100.00"
            )
        }

        widget.update_positions(positions, "100.00")

        mock_table.clear.assert_called_once()
        # Verify row content
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        # Quantity, Entry, Current, PnL, %, Leverage are now Text objects
        assert str(args[1]) == "0.5"
        assert str(args[2]) == "50000.00"
        assert str(args[3]) == "50000.00"  # Current defaults to entry if mark_price missing
        assert str(args[4]) == "100.00"
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
                quantity="0.1",
                price="50000.00",
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
        assert str(args[3]) == "50000.00"
        assert args[4] == "OPEN"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        trades = [
            Trade(
                trade_id="trd_123",
                symbol="BTC-USD",
                side="SELL",
                quantity="0.1",
                price="51000.00",
                order_id="ord_123",
                time="2023-01-01T12:00:00.000Z",
            )
        ]

        widget.update_trades(trades)

        mock_table.clear.assert_called_once()
        # Verify row content includes Order ID
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SELL" in args[1]
        assert str(args[2]) == "0.1"
        assert str(args[3]) == "51000.00"
        assert args[4] == "ord_123"  # Order ID (short enough to not be truncated)

    def test_update_trades_truncation(self):
        widget = TradesWidget()
        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        long_id = "12345678-1234-1234-1234-1234567890ab"
        trades = [
            Trade(
                trade_id="trd_123",
                symbol="BTC-USD",
                side="SELL",
                quantity="0.1",
                price="51000.00",
                order_id=long_id,
                time="2023-01-01T12:00:00.000Z",
            )
        ]

        widget.update_trades(trades)
        args, _ = mock_table.add_row.call_args
        assert args[4] == long_id[-8:]
        # Wait, my code was [-8:]. "1234567890ab" -> "34567890ab"? No.
        # "12345678-1234-1234-1234-1234567890ab"
        # Last 8: "7890ab" is 6 chars. "567890ab" is 8.
        # Let's check the ID again.
        # UUID usually ends with 12 chars.
        # Let's just assert it ends with the last 8 chars of the input.
        assert args[4] == long_id[-8:]
