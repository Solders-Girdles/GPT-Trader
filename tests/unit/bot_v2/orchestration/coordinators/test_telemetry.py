from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


def _make_context(
    *,
    broker: object | None = None,
    risk_manager: object | None = None,
    symbols: tuple[str, ...] = ("BTC-PERP",),
) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD)
    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=Mock(),
        orders_store=Mock(),
    )
    runtime_state = PerpsBotRuntimeState(list(symbols))

    return CoordinatorContext(
        config=config,
        registry=registry,
        event_store=registry.event_store,
        orders_store=registry.orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=symbols,
        bot_id="perps_bot",
        runtime_state=runtime_state,
    )


def test_initialize_without_broker() -> None:
    context = _make_context(broker=None)
    coordinator = TelemetryCoordinator(context)

    updated = coordinator.initialize(context)

    assert updated.registry.extras == {}


def test_initialize_with_broker() -> None:
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    risk_manager = Mock()
    context = _make_context(broker=broker, risk_manager=risk_manager)

    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)

    extras = updated.registry.extras
    assert "account_manager" in extras
    assert "account_telemetry" in extras
    assert "market_monitor" in extras


@pytest.mark.asyncio
async def test_start_background_tasks_starts_account_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = Mock()
    risk_manager = Mock()
    account_telemetry = Mock()
    account_telemetry.supports_snapshots.return_value = True
    account_telemetry.run = AsyncMock()

    context = _make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated_context = coordinator.initialize(context)

    extras = dict(updated_context.registry.extras)
    extras["account_telemetry"] = account_telemetry
    updated_context = updated_context.with_updates(
        registry=updated_context.registry.with_updates(extras=extras)
    )
    coordinator.update_context(updated_context)

    tasks = await coordinator.start_background_tasks()
    assert len(tasks) >= 1

    for task in tasks:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_shutdown_cancels_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.stream_orderbook.return_value = []
    risk_manager = Mock()
    risk_manager.last_mark_update = {}

    context = _make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)
    coordinator.update_context(
        updated.with_updates(
            config=updated.config.model_copy(update={"perps_enable_streaming": True})
        )
    )

    tasks = await coordinator.start_background_tasks()
    await coordinator.shutdown()

    for task in tasks:
        assert task.cancelled() or task.done()
