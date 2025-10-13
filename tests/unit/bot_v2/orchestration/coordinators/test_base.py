from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from bot_v2.orchestration.coordinators.base import (
    BaseCoordinator,
    CoordinatorContext,
)


def _make_context() -> CoordinatorContext:
    config = SimpleNamespace()
    registry = SimpleNamespace()
    return CoordinatorContext(
        config=config,
        registry=registry,
        symbols=("BTC-USD",),
        bot_id="test-bot",
        event_store=None,
        orders_store=None,
    )


def test_coordinator_context_with_updates() -> None:
    context = _make_context()
    updated = context.with_updates(broker="broker")

    assert updated is not context
    assert updated.broker == "broker"
    assert context.broker is None
    assert updated.symbols == context.symbols


class DummyCoordinator(BaseCoordinator):
    @property
    def name(self) -> str:  # pragma: no cover - trivial delegation
        return "dummy"


@pytest.mark.asyncio
async def test_base_coordinator_defaults() -> None:
    context = _make_context()
    coordinator = DummyCoordinator(context)

    result = coordinator.initialize(context)
    assert result is context

    tasks = await coordinator.start_background_tasks()
    assert tasks == []

    health = coordinator.health_check()
    assert health.healthy is True
    assert health.component == "dummy"
    assert health.details["background_tasks"] == 0

    await coordinator.shutdown()
    assert coordinator.health_check().details["shutdown_complete"] is True


class BackgroundTaskCoordinator(BaseCoordinator):
    @property
    def name(self) -> str:
        return "background"

    async def start_background_tasks(self):
        task = asyncio.create_task(asyncio.sleep(0))
        self._register_background_task(task)
        return [task]


@pytest.mark.asyncio
async def test_base_coordinator_background_task_cleanup() -> None:
    context = _make_context()
    coordinator = BackgroundTaskCoordinator(context)

    tasks = await coordinator.start_background_tasks()
    assert len(tasks) == 1
    await asyncio.sleep(0)

    details = coordinator.health_check().details
    assert details["background_tasks"] == 1

    await coordinator.shutdown()

    details = coordinator.health_check().details
    assert details["background_tasks"] == 0
