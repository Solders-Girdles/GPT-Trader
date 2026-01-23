from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpt_trader.tui.managers.ui_coordinator import UICoordinator


class _BrokerCallsSpy:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def __call__(self, func, *args, **kwargs):
        self.calls.append(func)
        return func(*args, **kwargs)


class _TestClient:
    def get_product_book(self, product_id: str, limit: int = 1) -> dict[str, list[dict[str, str]]]:
        return {
            "bids": [{"price": "100"}],
            "asks": [{"price": "101"}],
        }

    def get_resilience_status(self) -> dict[str, str]:
        return {"status": "ok"}


def _make_app(client: _TestClient, broker_calls: _BrokerCallsSpy) -> SimpleNamespace:
    broker = SimpleNamespace(_client=client)
    context = SimpleNamespace(broker=broker, broker_calls=broker_calls)
    engine = SimpleNamespace(context=context)
    bot = SimpleNamespace(engine=engine)
    market_data = SimpleNamespace(prices={"BTC-USD": Decimal("20000")}, spreads={})
    tui_state = SimpleNamespace(market_data=market_data, update_resilience_data=Mock())
    return SimpleNamespace(bot=bot, tui_state=tui_state)


@pytest.mark.asyncio
async def test_collect_spread_data_uses_broker_calls() -> None:
    broker_calls = _BrokerCallsSpy()
    app = _make_app(_TestClient(), broker_calls)
    coordinator = UICoordinator(app)

    await coordinator._collect_spread_data()

    assert broker_calls.calls
    assert app.tui_state.market_data.spreads["BTC-USD"] > 0


@pytest.mark.asyncio
async def test_collect_resilience_metrics_uses_broker_calls() -> None:
    broker_calls = _BrokerCallsSpy()
    app = _make_app(_TestClient(), broker_calls)
    coordinator = UICoordinator(app)

    await coordinator._collect_resilience_metrics()

    assert broker_calls.calls
    app.tui_state.update_resilience_data.assert_called_once_with({"status": "ok"})
