from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.widgets.positions import OrdersWidget, PositionsWidget, TradesWidget


class TestPositionsWidget:
    def test_update_positions(self):
        widget = PositionsWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        positions = {
            "BTC-USD": {"quantity": "0.5", "entry_price": "50000.00", "unrealized_pnl": "100.00"}
        }

        widget.update_positions(positions, "100.00")

        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_with("BTC-USD", "0.5", "50000.00", "100.00")


class TestOrdersWidget:
    def test_update_orders(self):
        widget = OrdersWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        orders = [
            {
                "symbol": "BTC-USD",
                "side": "BUY",
                "quantity": "0.1",
                "price": "50000.00",
                "status": "OPEN",
                "type": "LIMIT",
                "time_in_force": "GTC",
                "timestamp": 1600000000,
            }
        ]

        widget.update_orders(orders)

        mock_table.clear.assert_called_once()
        # Verify row content includes Type and TIF
        # Updated to match simplified columns: Symbol, Side, Quantity, Price, Status
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert args[2] == "0.1"
        assert args[3] == "50000.00"
        assert args[4] == "OPEN"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()

        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

        trades = [
            {
                "symbol": "BTC-USD",
                "side": "SELL",
                "quantity": "0.1",
                "price": "51000.00",
                "order_id": "ord_123",
                "timestamp": 1600000000,
            }
        ]

        widget.update_trades(trades)

        mock_table.clear.assert_called_once()
        # Verify row content includes Order ID
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SELL" in args[1]
        assert args[2] == "0.1"
        assert args[3] == "51000.00"
        assert args[4] == "ord_123"  # Order ID
