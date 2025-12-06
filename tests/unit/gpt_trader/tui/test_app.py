import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
)
from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.widgets import (
    BotStatusWidget,
    MarketWatchWidget,
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

        # First update (no previous) - now using Decimal
        widget.update_prices({"BTC": Decimal("100")}, 1000)
        # Should be white (default) or at least not crash
        args, _ = mock_table.add_row.call_args
        assert "BTC" in args

        # Second update (higher)
        widget.update_prices({"BTC": Decimal("110")}, 1001)
        args, _ = mock_table.add_row.call_args
        # Price should be formatted with commas
        assert "[#85B77F]" in str(args[1])  # Theme success (warm green)

        # Third update (lower)
        widget.update_prices({"BTC": Decimal("105")}, 1002)
        args, _ = mock_table.add_row.call_args
        assert "[#E08580]" in str(args[1])  # Theme error (warm coral-red)


class TestTuiLogHandler:
    def test_log_coloring(self):
        # Mock widget with required lifecycle properties
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler = TuiLogHandler()  # No arguments in new API
        handler.register_widget(mock_widget, min_level=logging.INFO)

        # Test INFO - handler should call widget.write() with markup + newline
        record = logging.LogRecord("name", logging.INFO, "path", 1, "Info message", (), None)
        handler.emit(record)
        assert mock_widget.write.called

        # Test ERROR
        record = logging.LogRecord("name", logging.ERROR, "path", 1, "Error message", (), None)
        handler.emit(record)
        # Verify write was called at least twice (once for INFO, once for ERROR)
        assert mock_widget.write.call_count >= 2


@pytest.mark.asyncio
async def test_app_instantiation(mock_bot):
    app = TraderApp(mock_bot)

    assert app.bot == mock_bot
    assert hasattr(app, "tui_state")
    assert app.tui_state is not None


@pytest.mark.asyncio
async def test_app_sync_state(mock_bot):
    mock_bot.running = True
    # Mock engine and status reporter with typed BotStatus
    mock_bot.engine.status_reporter.get_status = MagicMock(
        return_value=BotStatus(
            bot_id="test-bot",
            timestamp=1600000000.0,
            timestamp_iso="2020-09-13T12:26:40Z",
            version="test",
            engine=EngineStatus(),
            market=MarketStatus(),
            positions=PositionStatus(equity=Decimal("999.00")),
            orders=[],
            trades=[],
            account=AccountStatus(),
            strategy=StrategyStatus(),
            risk=RiskStatus(),
            system=SystemStatus(),
            heartbeat=HeartbeatStatus(),
        )
    )

    app = TraderApp(mock_bot)
    app._sync_state_from_bot()

    assert app.tui_state.running is True
    assert app.tui_state.position_data.equity == Decimal("999.00")
