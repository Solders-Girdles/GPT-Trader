"""Race coverage for emergency flatten while the bot run loop is active."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import gpt_trader.features.live_trade.bot as bot_module
from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.bot import TradingBot
from gpt_trader.features.live_trade.lifecycle import TradingBotState


class _DirectBrokerCalls:
    def __init__(self) -> None:
        self.shutdown = Mock()

    async def __call__(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _RunningEngine:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.shutdown_calls = 0
        self._task: asyncio.Task[None] | None = None

    async def start_background_tasks(self) -> list[asyncio.Task[None]]:
        self._task = asyncio.create_task(self._background_task())
        self.started.set()
        return [self._task]

    async def _background_task(self) -> None:
        await asyncio.Event().wait()

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_flatten_and_stop_preserves_error_when_run_cleanup_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def place_order(**kwargs):
        if kwargs["symbol"] == "ETH-USD":
            raise RuntimeError("venue rejected emergency close")
        return None

    config = BotConfig(symbols=["BTC-USD", "ETH-USD"], interval=1)
    broker = Mock()
    broker.list_positions.return_value = [
        SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1")),
        SimpleNamespace(symbol="ETH-USD", quantity=Decimal("-2")),
    ]
    broker.place_order = Mock(side_effect=place_order)
    event_store = Mock()
    notification_service = Mock()
    notification_service.notify = AsyncMock(return_value=True)

    container = SimpleNamespace(
        broker=broker,
        risk_manager=Mock(),
        event_store=event_store,
        orders_store=Mock(),
        notification_service=notification_service,
        health_state=Mock(),
    )

    engine = _RunningEngine()
    monkeypatch.setattr(bot_module, "TradingEngine", MagicMock(return_value=engine))

    bot = TradingBot(config=config, container=container)
    bot._broker_calls = _DirectBrokerCalls()

    run_task = asyncio.create_task(bot.run(single_cycle=False))
    await engine.started.wait()

    messages = await bot.flatten_and_stop()
    await asyncio.wait_for(run_task, timeout=1)

    assert any("Emergency flatten incomplete" in msg for msg in messages)
    assert engine.shutdown_calls >= 2
    assert bot.state is TradingBotState.ERROR
    assert bot._lifecycle.last_transition is not None
    assert bot._lifecycle.last_transition.reason == "flatten_and_stop_failed"
    bot._broker_calls.shutdown.assert_not_called()
