from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.widgets import AccountWidget, OrdersWidget, TradesWidget


class TestAccountWidget:
    def test_update_account(self):
        widget = AccountWidget()
        mock_table = MagicMock()
        mock_label = MagicMock()

        # Mock query_one to return table for DataTable and label for Label
        def query_side_effect(selector, type=None):
            if selector == DataTable or type == DataTable:
                return mock_table
            return mock_label

        widget.query_one = MagicMock(side_effect=query_side_effect)

        data = MagicMock()
        data.volume_30d = "1000.00"
        data.fees_30d = "5.00"
        data.fee_tier = "Taker"
        data.balances = [{"asset": "USD", "total": "100", "available": "50"}]

        widget.update_account(data)

        # Verify labels updated
        assert mock_label.update.call_count >= 3

        # Verify table updated
        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "USD"
        assert args[1] == "100"
        assert args[2] == "50"


class TestOrdersWidget:
    def test_update_orders(self):
        widget = OrdersWidget()
        mock_table = MagicMock()
        widget.query_one = MagicMock(return_value=mock_table)

        orders = [
            {
                "symbol": "BTC",
                "side": "BUY",
                "quantity": "1",
                "price": "100",
                "status": "OPEN",
                "timestamp": 1600000000,
            }
        ]

        widget.update_orders(orders)

        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC"
        assert "BUY" in args[1]
        assert args[2] == "1"
        assert args[3] == "100"
        assert args[4] == "OPEN"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()
        mock_table = MagicMock()
        widget.query_one = MagicMock(return_value=mock_table)

        trades = [
            {
                "symbol": "ETH",
                "side": "SELL",
                "quantity": "2",
                "price": "2000",
                "timestamp": 1600000000,
            }
        ]

        widget.update_trades(trades)

        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "ETH"
        assert "SELL" in args[1]
        assert args[2] == "2"
        assert args[3] == "2000"
