import logging
from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.widgets import BotStatusWidget, MarketWatchWidget


class TestBotStatusWidget:
    def test_equity_update(self):
        widget = BotStatusWidget()
        mock_label = MagicMock()
        widget.query_one = MagicMock(return_value=mock_label)

        widget.equity = "1000.50"
        assert widget.equity == "1000.50"
        mock_label.update.assert_called_with("$1000.50")


class TestMarketWatchWidget:
    def test_price_coloring(self):
        widget = MarketWatchWidget()
        mock_table = MagicMock()
        widget.query_one = MagicMock(return_value=mock_table)
        widget.previous_prices = {}

        widget.update_prices({"BTC": Decimal("100")}, 1000)
        args, _ = mock_table.add_row.call_args
        assert "BTC" in args

        widget.update_prices({"BTC": Decimal("110")}, 1001)
        args, _ = mock_table.add_row.call_args
        assert f"[{THEME.colors.success}]" in str(args[1])

        widget.update_prices({"BTC": Decimal("105")}, 1002)
        args, _ = mock_table.add_row.call_args
        assert f"[{THEME.colors.error}]" in str(args[1])


class TestTuiLogHandler:
    def test_log_coloring(self):
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler = TuiLogHandler()
        handler.register_widget(mock_widget, min_level=logging.INFO)

        record = logging.LogRecord("name", logging.INFO, "path", 1, "Info message", (), None)
        handler.emit(record)
        assert mock_widget.write.called

        record = logging.LogRecord("name", logging.ERROR, "path", 1, "Error message", (), None)
        handler.emit(record)
        assert mock_widget.write.call_count >= 2
