import logging
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance
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
from gpt_trader.tui.theme import THEME
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
        assert f"[{THEME.colors.success}]" in str(args[1])

        # Third update (lower)
        widget.update_prices({"BTC": Decimal("105")}, 1002)
        args, _ = mock_table.add_row.call_args
        assert f"[{THEME.colors.error}]" in str(args[1])


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


@dataclass(frozen=True)
class _BootstrapConfig:
    symbols: list[str]


class _BootstrapBroker:
    def __init__(self) -> None:
        self._tickers = {
            "BTC-USD": {"price": "20000"},
            "ETH-USD": {"price": "1000"},
        }

    def list_balances(self) -> list[Balance]:
        return [
            Balance(asset="USD", total=Decimal("100"), available=Decimal("100")),
            Balance(asset="BTC", total=Decimal("0.5"), available=Decimal("0.5")),
        ]

    def get_ticker(self, product_id: str) -> dict:
        return self._tickers.get(product_id, {"price": "0"})


class _BootstrapEngine:
    def __init__(self) -> None:
        from gpt_trader.monitoring.status_reporter import StatusReporter

        self.status_reporter = StatusReporter()
        self.context = SimpleNamespace(runtime_state=None)


class _BootstrapBot:
    def __init__(self) -> None:
        self.running = False
        self.config = _BootstrapConfig(symbols=["BTC-USD", "ETH-USD"])
        self.broker = _BootstrapBroker()
        self.engine = _BootstrapEngine()


@pytest.mark.asyncio
async def test_bootstrap_snapshot_populates_balances_and_equity():
    bot = _BootstrapBot()
    app = TraderApp(bot=bot)
    app.data_source_mode = "paper"
    app.tui_state.data_source_mode = "paper"

    ok = await app.bootstrap_snapshot()
    assert ok is True

    # Balances should show up even while STOPPED.
    assert any(b.asset == "USD" for b in app.tui_state.account_data.balances)
    assert any(b.asset == "BTC" for b in app.tui_state.account_data.balances)

    # Equity is estimated from balances + tickers: 100 + 0.5 * 20000 = 10100
    assert app.tui_state.position_data.equity == Decimal("10100")

    # Market snapshot should include configured symbols.
    assert app.tui_state.market_data.prices.get("BTC-USD") == Decimal("20000")


class TestResourceCleanup:
    """Tests for proper resource cleanup on TUI shutdown."""

    def test_coinbase_client_has_close_method(self):
        """Verify CoinbaseClientBase has close() for explicit cleanup."""
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        client = CoinbaseClientBase()
        assert hasattr(client, "close")
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")

        # Should work without error
        client.close()

    def test_coinbase_client_context_manager(self):
        """Verify CoinbaseClientBase works as context manager."""
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        with CoinbaseClientBase() as client:
            assert client.session is not None

        # Session should still exist but be closed after context
        # (requests.Session.close() doesn't set to None)

    def test_websocket_has_close_method(self):
        """Verify CoinbaseWebSocket has close() for explicit cleanup."""
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        assert hasattr(ws, "close")
        assert hasattr(ws, "__enter__")
        assert hasattr(ws, "__exit__")
        assert hasattr(ws, "_shutdown")

        # Should work without error
        ws.close()

    def test_websocket_close_is_idempotent(self):
        """Verify calling close() multiple times is safe."""
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        ws.close()
        ws.close()  # Should not raise

    def test_performance_service_cleanup(self):
        """Verify performance service can be cleared."""
        from gpt_trader.tui.services.performance_service import (
            TuiPerformanceService,
            clear_tui_performance_service,
            get_tui_performance_service,
            set_tui_performance_service,
        )

        # Set up a service
        service = TuiPerformanceService(enabled=False)
        set_tui_performance_service(service)
        assert get_tui_performance_service() is service

        # Clear it
        clear_tui_performance_service()

        # Get should create a new one
        new_service = get_tui_performance_service()
        assert new_service is not service

        # Clean up for other tests
        clear_tui_performance_service()

    def test_app_cleanup_bot_resources_closes_client(self, mock_bot):
        """Verify _cleanup_bot_resources() calls close() on client."""
        # Add mock client to bot
        mock_client = MagicMock()
        mock_bot.client = mock_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_client.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_websocket(self, mock_bot):
        """Verify _cleanup_bot_resources() calls close() on websocket."""
        # Add mock websocket to bot
        mock_ws = MagicMock()
        mock_bot.websocket = mock_ws

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_ws.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_engine_client(self, mock_bot):
        """Verify _cleanup_bot_resources() calls close() on engine client."""
        # Add mock client to engine
        mock_engine_client = MagicMock()
        mock_bot.engine.client = mock_engine_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_engine_client.close.assert_called_once()

    def test_app_cleanup_handles_missing_resources(self, mock_bot):
        """Verify _cleanup_bot_resources() handles bots without client/ws."""
        # Remove any client/websocket attributes
        if hasattr(mock_bot, "client"):
            del mock_bot.client
        if hasattr(mock_bot, "websocket"):
            del mock_bot.websocket

        app = TraderApp(mock_bot)
        # Should not raise
        app._cleanup_bot_resources()

    def test_app_cleanup_handles_close_errors(self, mock_bot):
        """Verify _cleanup_bot_resources() handles errors gracefully."""
        # Add client that raises on close
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Close failed")
        mock_bot.client = mock_client

        app = TraderApp(mock_bot)
        # Should not raise, just log warning
        app._cleanup_bot_resources()

    def test_websocket_reconnect_limit_configurable(self):
        """Verify WebSocket reconnect limit can be configured via constants."""
        from gpt_trader.config.constants import MAX_WS_RECONNECT_ATTEMPTS

        # Default is 50
        assert MAX_WS_RECONNECT_ATTEMPTS == 10

        # Can be set to 0 for unlimited (via env var in real usage)
        # Just verify the constant exists and is accessible
