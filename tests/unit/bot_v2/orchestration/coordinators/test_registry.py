from __future__ import annotations

import asyncio
from typing import List

import pytest

from bot_v2.orchestration.coordinators.base import BaseCoordinator, CoordinatorContext
from bot_v2.orchestration.coordinators.registry import CoordinatorRegistry


def _make_context() -> CoordinatorContext:
    config = object()
    registry = object()
    return CoordinatorContext(
        config=config, registry=registry, symbols=(), event_store=None, orders_store=None
    )


class InitCoordinator(BaseCoordinator):
    def __init__(self, context: CoordinatorContext, name: str, broker_value: str):
        super().__init__(context)
        self._name = name
        self._broker_value = broker_value

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        return context.with_updates(broker=self._broker_value)


class ObserverCoordinator(BaseCoordinator):
    def __init__(self, context: CoordinatorContext, name: str):
        super().__init__(context)
        self._name = name
        self.received_contexts: list[CoordinatorContext] = []

    @property
    def name(self) -> str:
        return self._name

    def update_context(self, context: CoordinatorContext) -> None:
        super().update_context(context)
        self.received_contexts.append(context)


def test_registry_initialization_updates_all_coordinators() -> None:
    context = _make_context()
    registry = CoordinatorRegistry(context)
    initiator = InitCoordinator(context, "initiator", "broker-1")
    observer = ObserverCoordinator(context, "observer")

    registry.register(initiator)
    registry.register(observer)

    final_context = registry.initialize_all()

    assert final_context.broker == "broker-1"
    assert observer.received_contexts[-1].broker == "broker-1"


def test_registry_rejects_duplicate_registration() -> None:
    context = _make_context()
    registry = CoordinatorRegistry(context)
    coordinator = InitCoordinator(context, "duplicate", "broker")

    registry.register(coordinator)
    with pytest.raises(ValueError):
        registry.register(coordinator)


class TaskCoordinator(BaseCoordinator):
    def __init__(self, context: CoordinatorContext, name: str, recorder: list[str]):
        super().__init__(context)
        self._name = name
        self._recorder = recorder

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        self._recorder.append(f"init-{self._name}")
        return context

    async def start_background_tasks(self):
        self._recorder.append(f"start-{self._name}")
        task = asyncio.create_task(asyncio.sleep(0))
        self._register_background_task(task)
        return [task]

    async def shutdown(self) -> None:
        self._recorder.append(f"shutdown-{self._name}")
        await super().shutdown()


@pytest.mark.asyncio
async def test_registry_manages_lifecycle_order() -> None:
    context = _make_context()
    registry = CoordinatorRegistry(context)
    order: list[str] = []

    first = TaskCoordinator(context, "first", order)
    second = TaskCoordinator(context, "second", order)

    registry.register(first)
    registry.register(second)

    registry.initialize_all()
    tasks = await registry.start_all_background_tasks()

    await asyncio.sleep(0)
    assert order[:2] == ["init-first", "init-second"]
    assert order[2:4] == ["start-first", "start-second"]
    assert len(tasks) == 2

    await registry.shutdown_all()

    assert order[4:] == ["shutdown-second", "shutdown-first"]

    health = registry.health_check_all()
    assert set(health.keys()) == {"first", "second"}
    assert all(status.healthy is False for status in health.values())
