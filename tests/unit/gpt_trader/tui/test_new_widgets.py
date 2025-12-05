from decimal import Decimal
from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import AccountBalance, AccountSummary, Order, Trade
from gpt_trader.tui.widgets import AccountWidget, OrdersWidget, TradesWidget


class TestAccountWidget:
    def test_update_account(self):
        widget = AccountWidget(compact_mode=False)
        mock_table = MagicMock()
        mock_label = MagicMock()

        # Mock query_one to return table for DataTable and label for Label
        def query_side_effect(selector, type=None):
            if selector == DataTable or type == DataTable:
                return mock_table
            return mock_label

        widget.query_one = MagicMock(side_effect=query_side_effect)

        data = AccountSummary(
            volume_30d=Decimal("1000.00"),
            fees_30d=Decimal("5.00"),
            fee_tier="Taker",
            balances=[
                AccountBalance(
                    asset="USD", total=Decimal("100"), available=Decimal("50"), hold=Decimal("0")
                )
            ],
        )

        widget.update_account(data)

        # Verify labels updated (at least 2 calls expected)
        assert mock_label.update.call_count >= 2

        # Verify table updated
        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "USD"
        # Values are formatted with commas, so check for string representation
        assert "100" in str(args[1])
        assert "50" in str(args[2])


class TestOrdersWidget:
    def test_update_orders(self):
        widget = OrdersWidget()
        mock_table = MagicMock()
        widget.query_one = MagicMock(return_value=mock_table)

        orders = [
            Order(
                order_id="ord_1",
                symbol="BTC",
                side="BUY",
                quantity=Decimal("1"),
                price=Decimal("100"),
                status="OPEN",
                creation_time="1600000000",
            )
        ]

        widget.update_orders(orders)

        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC"
        assert "BUY" in str(args[1])
        # Quantity and price are formatted, check string representation
        assert "1" in str(args[2])
        assert "100" in str(args[3])
        assert args[4] == "OPEN"


class TestTradesWidget:
    def test_update_trades(self):
        widget = TradesWidget()
        mock_table = MagicMock()
        mock_label = MagicMock()

        # Mock query_one to return table for DataTable and label for Label
        def query_side_effect(selector, type=None):
            if "#trades-table" in str(selector):
                return mock_table
            return mock_label

        widget.query_one = MagicMock(side_effect=query_side_effect)

        # Mock the trade matcher
        mock_matcher = MagicMock()
        mock_matcher.process_trades.return_value = {"trd_1": "+10.50"}
        widget._trade_matcher = mock_matcher

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="ETH",
                side="SELL",
                quantity=Decimal("2"),
                price=Decimal("2000"),
                order_id="ord_2",
                time="2023-01-01T12:00:00.000Z",
                fee=Decimal("0"),
            )
        ]

        widget.update_trades(trades)

        mock_table.clear.assert_called_once()
        mock_table.add_row.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "ETH"
        assert "SELL" in str(args[1])
        # Quantity and price are formatted, check string representation
        assert "2" in str(args[2])
        assert "2000" in str(args[3]) or "2,000" in str(args[3])
