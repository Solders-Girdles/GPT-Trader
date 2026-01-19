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
