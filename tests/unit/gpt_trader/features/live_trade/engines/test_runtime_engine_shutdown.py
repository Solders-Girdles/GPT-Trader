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


def _context() -> CoordinatorContext:
    return CoordinatorContext(config=Mock(symbols=()), container=object())


@pytest.mark.asyncio
async def test_runtime_engine_concurrent_shutdown_awaits_in_flight_drain() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.2)
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
                timeout_seconds=0.2,
                register_task=False,
            ),
        ),
        shutdown_step_timeout_seconds=0.2,
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
            timeout=0.5,
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
