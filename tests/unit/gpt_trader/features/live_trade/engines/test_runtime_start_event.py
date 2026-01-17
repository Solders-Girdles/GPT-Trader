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


@pytest.fixture
def engine() -> Iterator[TradingEngine]:
    risk = BotRiskConfig()
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    config.profile = "dev"
    container = ApplicationContainer(config)
    set_application_container(container)
    engine_context = CoordinatorContext(
        config=config,
        broker=None,
        risk_manager=None,
        event_store=container.event_store,
        orders_store=container.orders_store,
        container=container,
    )
    engine = TradingEngine(engine_context)
    try:
        yield engine
    finally:
        clear_application_container()


@pytest.mark.asyncio
async def test_runtime_start_event_records_null_build_sha(engine: TradingEngine) -> None:
    engine._event_store = MagicMock()

    engine._heartbeat.start = AsyncMock(return_value=None)
    engine._status_reporter.start = AsyncMock(return_value=None)
    engine._system_maintenance.start_prune_loop = AsyncMock(
        return_value=asyncio.create_task(asyncio.sleep(0))
    )

    engine._health_check_runner = MagicMock()
    engine._health_check_runner.start = AsyncMock()

    async def mock_run_loop() -> None:
        return None

    async def mock_monitor_ws_health() -> None:
        return None

    engine._run_loop = mock_run_loop
    engine._monitor_ws_health = mock_monitor_ws_health
    engine._should_enable_streaming = lambda: False

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("GPT_TRADER_BUILD_SHA", "")
        monkeypatch.delenv("GPT_TRADER_BUILD_SHA", raising=False)
        engine.running = False
        await engine.start_background_tasks()

    assert engine._event_store.append.called
    event_type, payload = engine._event_store.append.call_args[0]
    assert event_type == "runtime_start"
    assert payload["build_sha"] is None
