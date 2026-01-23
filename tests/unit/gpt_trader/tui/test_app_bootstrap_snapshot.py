from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import pytest

from gpt_trader.core import Balance
from gpt_trader.tui.app import TraderApp


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


class _BrokerCallsSpy:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def __call__(self, func, *args, **kwargs):
        self.calls.append(func)
        return func(*args, **kwargs)


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


@pytest.mark.asyncio
async def test_bootstrap_snapshot_uses_broker_calls():
    bot = _BootstrapBot()
    broker_calls = _BrokerCallsSpy()
    bot.context = SimpleNamespace(broker_calls=broker_calls)
    app = TraderApp(bot=bot)
    app.data_source_mode = "paper"
    app.tui_state.data_source_mode = "paper"

    ok = await app.bootstrap_snapshot()
    assert ok is True
    assert len(broker_calls.calls) == 1
