from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.runtime import (
    RuntimeEngine,
    RuntimeEngineState,
    RuntimeLifecycleError,
    RuntimeLifecyclePlan,
    RuntimeLifecycleStep,
    RuntimeStepKind,
)


def _context() -> CoordinatorContext:
    return CoordinatorContext(config=Mock(symbols=()), container=object())


@pytest.mark.asyncio
async def test_runtime_engine_startup_timeout_does_not_wait_for_cancel_cleanup() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.01)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def cancellation_resistant_startup() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep(
                "slow_startup",
                RuntimeStepKind.STARTUP_HOOK,
                cancellation_resistant_startup,
                timeout_seconds=0.01,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=0.01,
    )

    try:
        with pytest.raises(RuntimeLifecycleError, match="slow_startup"):
            await asyncio.wait_for(runtime.start(plan), timeout=0.2)

        await asyncio.wait_for(cleanup_started.wait(), timeout=0.2)

        assert runtime.state == RuntimeEngineState.FAILED
        assert any(
            event.name == "slow_startup"
            and event.success is False
            and event.error is not None
            and "TimeoutError" in event.error
            for event in runtime.events
        )
        assert all(event.name != "state_running" for event in runtime.events)
    finally:
        release_cleanup.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_runtime_engine_startup_cancellation_drains_registered_tasks() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.2)
    startup_started = asyncio.Event()
    events: list[str] = []
    created_tasks: list[asyncio.Task[None]] = []

    async def worker() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            events.append("worker:cancelled")
            raise

    def task_factory() -> asyncio.Task[None]:
        task = asyncio.create_task(worker(), name="startup_registered_worker")
        created_tasks.append(task)
        return task

    async def slow_startup() -> None:
        events.append("startup_waiting")
        startup_started.set()
        await asyncio.sleep(60)

    async def shutdown_hook() -> None:
        events.append("shutdown")

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
            RuntimeLifecycleStep(
                "slow_startup",
                RuntimeStepKind.STARTUP_HOOK,
                slow_startup,
                timeout_seconds=0.2,
                register_task=False,
            ),
        ),
        shutdown_steps=(
            RuntimeLifecycleStep(
                "shutdown",
                RuntimeStepKind.SHUTDOWN_HOOK,
                shutdown_hook,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=0.2,
    )

    start_task = asyncio.create_task(runtime.start(plan))
    try:
        await startup_started.wait()

        start_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=2.0)

        assert runtime.state == RuntimeEngineState.FAILED
        assert runtime.last_error == "startup cancelled"
        assert runtime.background_tasks == []
        assert created_tasks and created_tasks[0].cancelled()
        assert events == ["startup_waiting", "shutdown", "worker:cancelled"]
    finally:
        if not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
