from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gpt_trader.tui.app_lifecycle as app_lifecycle_module
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


@pytest.mark.asyncio
async def test_app_instantiation(mock_bot):
    app = TraderApp(mock_bot)

    assert app.bot == mock_bot
    assert hasattr(app, "tui_state")
    assert app.tui_state is not None


@pytest.mark.asyncio
async def test_app_sync_state(mock_bot):
    mock_bot.running = True
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

    assert any(b.asset == "USD" for b in app.tui_state.account_data.balances)
    assert any(b.asset == "BTC" for b in app.tui_state.account_data.balances)

    assert app.tui_state.position_data.equity == Decimal("10100")

    assert app.tui_state.market_data.prices.get("BTC-USD") == Decimal("20000")


class TestResourceCleanup:
    """Tests for proper resource cleanup on TUI shutdown."""

    def test_coinbase_client_has_close_method(self):
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        client = CoinbaseClientBase()
        assert hasattr(client, "close")
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")

        client.close()

    def test_coinbase_client_context_manager(self):
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        with CoinbaseClientBase() as client:
            assert client.session is not None

    def test_websocket_has_close_method(self):
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        assert hasattr(ws, "close")
        assert hasattr(ws, "__enter__")
        assert hasattr(ws, "__exit__")
        assert hasattr(ws, "_shutdown")

        ws.close()

    def test_websocket_close_is_idempotent(self):
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        ws.close()
        ws.close()
        assert ws._closed is True
        assert ws._shutdown.is_set() is True
        assert ws.ws is None
        assert ws.wst is None

    def test_performance_service_cleanup(self):
        from gpt_trader.tui.services.performance_service import (
            TuiPerformanceService,
            clear_tui_performance_service,
            get_tui_performance_service,
            set_tui_performance_service,
        )

        service = TuiPerformanceService(enabled=False)
        set_tui_performance_service(service)
        assert get_tui_performance_service() is service

        clear_tui_performance_service()

        new_service = get_tui_performance_service()
        assert new_service is not service

        clear_tui_performance_service()

    def test_app_cleanup_bot_resources_closes_client(self, mock_bot):
        mock_client = MagicMock()
        mock_bot.client = mock_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_client.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_websocket(self, mock_bot):
        mock_ws = MagicMock()
        mock_bot.websocket = mock_ws

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_ws.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_engine_client(self, mock_bot):
        mock_engine_client = MagicMock()
        mock_bot.engine.client = mock_engine_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_engine_client.close.assert_called_once()

    def test_app_cleanup_handles_missing_resources(self, mock_bot):
        if hasattr(mock_bot, "client"):
            del mock_bot.client
        if hasattr(mock_bot, "websocket"):
            del mock_bot.websocket

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()
        assert not hasattr(mock_bot, "client")
        assert not hasattr(mock_bot, "websocket")

    def test_app_cleanup_handles_close_errors(self, mock_bot, monkeypatch: pytest.MonkeyPatch):
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Close failed")
        mock_bot.client = mock_client
        mock_logger = MagicMock()
        monkeypatch.setattr(app_lifecycle_module, "logger", mock_logger)

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_logger.warning.assert_called_once()

    def test_websocket_reconnect_limit_configurable(self):
        from gpt_trader.config.constants import MAX_WS_RECONNECT_ATTEMPTS

        assert MAX_WS_RECONNECT_ATTEMPTS == 10
