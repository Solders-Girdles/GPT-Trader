from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.runtime import (
    RuntimeEngine,
    RuntimeEngineState,
    RuntimeLifecyclePlan,
    RuntimeLifecycleStep,
    RuntimeStepKind,
)

_TEST_LIFECYCLE_TIMEOUT = 5.0


def _context() -> CoordinatorContext:
    return CoordinatorContext(config=Mock(symbols=()), container=object())


@pytest.mark.asyncio
async def test_runtime_engine_concurrent_shutdown_awaits_in_flight_drain() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=_TEST_LIFECYCLE_TIMEOUT)
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    shutdown_calls = 0

    async def shutdown_hook() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1
        shutdown_started.set()
        await release_shutdown.wait()

    plan = RuntimeLifecyclePlan(
        shutdown_steps=(
            RuntimeLifecycleStep(
                "shutdown",
                RuntimeStepKind.SHUTDOWN_HOOK,
                shutdown_hook,
                timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
                register_task=False,
            ),
        ),
        shutdown_step_timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
    )

    first_shutdown: asyncio.Task[None] | None = None
    second_shutdown: asyncio.Task[None] | None = None
    try:
        await runtime.start(plan)
        first_shutdown = asyncio.create_task(runtime.shutdown(plan))
        await shutdown_started.wait()

        second_shutdown = asyncio.create_task(runtime.shutdown(plan))
        await asyncio.sleep(0)

        assert not second_shutdown.done()

        release_shutdown.set()
        await asyncio.wait_for(
            asyncio.gather(first_shutdown, second_shutdown),
            timeout=2.0,
        )
    finally:
        release_shutdown.set()
        pending_shutdowns = [
            task
            for task in (first_shutdown, second_shutdown)
            if task is not None and not task.done()
        ]
        if pending_shutdowns:
            await asyncio.gather(*pending_shutdowns, return_exceptions=True)

    assert shutdown_calls == 1
    assert runtime.state == RuntimeEngineState.TERMINATED


@pytest.mark.asyncio
async def test_runtime_engine_shutdown_waits_for_initializing_startup() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=_TEST_LIFECYCLE_TIMEOUT)
    startup_started = asyncio.Event()
    release_startup = asyncio.Event()
    events: list[str] = []
    created_tasks: list[asyncio.Task[None]] = []

    async def slow_startup() -> None:
        events.append("startup_started")
        startup_started.set()
        await release_startup.wait()
        events.append("startup_finished")

    async def worker() -> None:
        await asyncio.sleep(60)

    def task_factory() -> asyncio.Task[None]:
        events.append("task_factory")
        task = asyncio.create_task(worker(), name="startup_worker")
        created_tasks.append(task)
        return task

    async def shutdown_hook() -> None:
        events.append("shutdown")

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep(
                "slow_startup",
                RuntimeStepKind.STARTUP_HOOK,
                slow_startup,
                timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
                register_task=False,
            ),
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
        ),
        shutdown_steps=(
            RuntimeLifecycleStep(
                "shutdown",
                RuntimeStepKind.SHUTDOWN_HOOK,
                shutdown_hook,
                timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
        shutdown_step_timeout_seconds=_TEST_LIFECYCLE_TIMEOUT,
    )

    start_task: asyncio.Task | None = None
    shutdown_task: asyncio.Task[None] | None = None
    try:
        start_task = asyncio.create_task(runtime.start(plan))
        await startup_started.wait()

        shutdown_task = asyncio.create_task(runtime.shutdown(plan))
        await asyncio.sleep(0)

        assert not shutdown_task.done()
        assert runtime.state == RuntimeEngineState.INITIALIZING
        assert events == ["startup_started"]

        release_startup.set()
        await asyncio.wait_for(
            asyncio.gather(start_task, shutdown_task),
            timeout=2.0,
        )
    finally:
        release_startup.set()
        pending_tasks = [
            task for task in (start_task, shutdown_task) if task is not None and not task.done()
        ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if runtime.state == RuntimeEngineState.RUNNING:
            await runtime.shutdown(plan)

    assert events == ["startup_started", "startup_finished", "task_factory", "shutdown"]
    assert created_tasks and created_tasks[0].cancelled()
    assert runtime.background_tasks == []
    assert runtime.state == RuntimeEngineState.TERMINATED
