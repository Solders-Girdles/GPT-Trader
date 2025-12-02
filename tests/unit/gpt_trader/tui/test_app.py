import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.widgets import (
    BlockChartWidget,
    BotStatusWidget,
    MarketWatchWidget,
    TuiLogHandler,
)


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
        widget.previous_prices = {}  # Simulate mount

        # First update (no previous)
        widget.update_prices({"BTC": "100"}, 1000)
        # Should be white (default) or at least not crash
        args, _ = mock_table.add_row.call_args
        assert "BTC" in args

        # Second update (higher)
        widget.update_prices({"BTC": "110"}, 1001)
        args, _ = mock_table.add_row.call_args
        assert "[#7AA874]110[/" in args[1]  # Claude Code success (warm green)

        # Third update (lower)
        widget.update_prices({"BTC": "105"}, 1002)
        args, _ = mock_table.add_row.call_args
        assert "[#D4736E]105[/" in args[1]  # Claude Code error (warm coral-red)


class TestBlockChartWidget:
    def test_chart_color_green(self):
        widget = BlockChartWidget()
        mock_static = MagicMock()
        widget.query_one = MagicMock(return_value=mock_static)

        prices = [Decimal("100"), Decimal("110")]
        widget.update_chart(prices)

        args, _ = mock_static.update.call_args
        assert "[#7AA874]" in args[0]  # Claude Code success (warm green)

    def test_chart_color_red(self):
        widget = BlockChartWidget()
        mock_static = MagicMock()
        widget.query_one = MagicMock(return_value=mock_static)

        prices = [Decimal("100"), Decimal("90")]
        widget.update_chart(prices)

        args, _ = mock_static.update.call_args
        assert "[#D4736E]" in args[0]  # Claude Code error (warm coral-red)


class TestTuiLogHandler:
    def test_log_coloring(self):
        mock_widget = MagicMock()
        handler = TuiLogHandler(mock_widget)

        # Test INFO
        record = logging.LogRecord("name", logging.INFO, "path", 1, "Info message", (), None)
        handler.emit(record)
        # Note: LogHandler might add additional parameters beyond just the message
        assert mock_widget.write_log.called
        call_args = mock_widget.write_log.call_args[0][0]
        assert "[#a3be8c]Info message[/#a3be8c]" in call_args  # Still uses Nord colors internally

        # Test ERROR
        record = logging.LogRecord("name", logging.ERROR, "path", 1, "Error message", (), None)
        handler.emit(record)
        assert mock_widget.write_log.called
        call_args = mock_widget.write_log.call_args[0][0]
        assert "[#bf616a]Error message[/#bf616a]" in call_args  # Still uses Nord colors internally


@pytest.mark.asyncio
async def test_app_instantiation(mock_bot):
    app = TraderApp(mock_bot)

    assert app.bot == mock_bot
    assert hasattr(app, "tui_state")
    assert app.tui_state is not None


@pytest.mark.asyncio
async def test_app_sync_state(mock_bot):
    mock_bot.running = True
    # Mock engine and status reporter
    mock_bot.engine.status_reporter.get_status = MagicMock(
        return_value={"positions": {"equity": "999.00"}}
    )

    app = TraderApp(mock_bot)
    app._sync_state_from_bot()

    assert app.tui_state.running is True
    assert app.tui_state.position_data.equity == "999.00"
