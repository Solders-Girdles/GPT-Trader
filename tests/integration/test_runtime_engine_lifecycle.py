from __future__ import annotations

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine

pytestmark = pytest.mark.integration


async def _sleep_until_cancelled() -> None:
    await asyncio.sleep(60)


@pytest.fixture
def engine() -> Iterator[TradingEngine]:
    config = BotConfig(
        symbols=["BTC-USD"],
        interval=0.01,
        risk=BotRiskConfig(),
    )
    config.profile = "dev"
    container = ApplicationContainer(config)
    set_application_container(container)
    engine_context = CoordinatorContext(
        config=config,
        broker=MagicMock(),
        risk_manager=None,
        event_store=container.event_store,
        orders_store=container.orders_store,
        container=container,
    )
    try:
        yield TradingEngine(engine_context)
    finally:
        clear_application_container()


@pytest.mark.asyncio
async def test_trading_engine_runtime_start_shutdown_are_idempotent(
    engine: TradingEngine,
) -> None:
    engine._event_store = MagicMock()

    async def run_loop() -> None:
        await _sleep_until_cancelled()

    async def ws_health() -> None:
        await _sleep_until_cancelled()

    engine._run_loop = run_loop
    engine._monitor_ws_health = ws_health
    engine._should_enable_streaming = lambda: False
    engine._health_check_runner.start = AsyncMock()
    engine._health_check_runner.stop = AsyncMock()
    engine._heartbeat.start = AsyncMock(
        return_value=asyncio.create_task(_sleep_until_cancelled(), name="heartbeat")
    )
    engine._heartbeat.stop = AsyncMock()
    engine._status_reporter.start = AsyncMock(
        return_value=asyncio.create_task(_sleep_until_cancelled(), name="status_reporter")
    )
    engine._status_reporter.stop = AsyncMock()
    engine._system_maintenance.start_prune_loop = AsyncMock(
        return_value=asyncio.create_task(
            _sleep_until_cancelled(),
            name="system_maintenance_prune",
        )
    )
    engine._system_maintenance.stop = AsyncMock()

    try:
        first_tasks = await engine.start_background_tasks()
        second_tasks = await engine.start_background_tasks()

        assert first_tasks == second_tasks
        assert len(first_tasks) == 5
        assert engine._event_store.append.call_count == 1
        engine._health_check_runner.start.assert_awaited_once()
        engine._heartbeat.start.assert_awaited_once()
        engine._status_reporter.start.assert_awaited_once()
        engine._system_maintenance.start_prune_loop.assert_awaited_once()
    finally:
        await engine.shutdown()
    await engine.shutdown()

    engine._health_check_runner.stop.assert_awaited_once()
    engine._heartbeat.stop.assert_awaited_once()
    engine._status_reporter.stop.assert_awaited_once()
    engine._system_maintenance.stop.assert_awaited_once()
    assert engine.running is False
    assert engine._background_tasks == []
