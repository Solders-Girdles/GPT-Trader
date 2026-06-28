from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.runtime import (
    RuntimeDependency,
    RuntimeEngine,
    RuntimeEngineState,
    RuntimeLifecycleError,
    RuntimeLifecyclePlan,
    RuntimeLifecycleStep,
    RuntimeStepKind,
    RuntimeStopCondition,
    RuntimeStopRequested,
)


def _context() -> CoordinatorContext:
    return CoordinatorContext(config=Mock(symbols=()), container=object())


async def _sleep_until_cancelled(events: list[str], name: str) -> None:
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        events.append(f"{name}:cancelled")
        raise


@pytest.mark.asyncio
async def test_runtime_engine_runs_lifecycle_steps_in_order_and_cleans_tasks() -> None:
    events: list[str] = []
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.2)
    registered: list[asyncio.Task] = []

    def startup() -> None:
        events.append("startup")

    def task_factory() -> asyncio.Task:
        events.append("task_factory")
        return asyncio.create_task(
            _sleep_until_cancelled(events, "worker"),
            name="worker",
        )

    async def health_hook() -> None:
        events.append("health")

    def policy_checkpoint() -> None:
        events.append("policy")

    async def shutdown_hook() -> None:
        events.append("shutdown")

    plan = RuntimeLifecyclePlan(
        dependencies=(RuntimeDependency("config", object()),),
        startup_steps=(
            RuntimeLifecycleStep("startup", RuntimeStepKind.STARTUP_HOOK, startup),
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
            RuntimeLifecycleStep("health", RuntimeStepKind.HEALTH, health_hook),
            RuntimeLifecycleStep(
                "policy",
                RuntimeStepKind.POLICY_CHECKPOINT,
                policy_checkpoint,
                register_task=False,
            ),
        ),
        shutdown_steps=(
            RuntimeLifecycleStep(
                "shutdown",
                RuntimeStepKind.SHUTDOWN_HOOK,
                shutdown_hook,
                timeout_seconds=0.2,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=0.2,
    )

    tasks = await runtime.start(plan, register_task=registered.append)
    await asyncio.sleep(0)

    assert runtime.state == RuntimeEngineState.RUNNING
    assert [task.get_name() for task in tasks] == ["worker"]
    assert tasks == registered
    assert events == ["startup", "task_factory", "health", "policy"]

    await runtime.shutdown(plan)

    assert runtime.state == RuntimeEngineState.TERMINATED
    assert events == [
        "startup",
        "task_factory",
        "health",
        "policy",
        "shutdown",
        "worker:cancelled",
    ]
    assert runtime.background_tasks == []


@pytest.mark.asyncio
async def test_runtime_engine_start_and_shutdown_are_idempotent() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.2)
    start_calls = 0
    shutdown_calls = 0

    async def worker() -> None:
        await asyncio.sleep(60)

    def task_factory() -> asyncio.Task:
        nonlocal start_calls
        start_calls += 1
        return asyncio.create_task(worker(), name="idempotent_worker")

    async def shutdown_hook() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
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

    first_tasks = await runtime.start(plan)
    second_tasks = await runtime.start(plan)

    assert first_tasks == second_tasks
    assert start_calls == 1

    await runtime.shutdown(plan)
    await runtime.shutdown(plan)

    assert shutdown_calls == 1
    assert runtime.state == RuntimeEngineState.TERMINATED


@pytest.mark.asyncio
async def test_runtime_engine_startup_failure_runs_shutdown_and_cleans_tasks() -> None:
    events: list[str] = []
    created_tasks: list[asyncio.Task] = []
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.2)

    def task_factory() -> asyncio.Task:
        task = asyncio.create_task(
            _sleep_until_cancelled(events, "failing_worker"),
            name="failing_worker",
        )
        created_tasks.append(task)
        return task

    def failing_step() -> None:
        events.append("failing_step")
        raise ValueError("boom")

    async def shutdown_hook() -> None:
        events.append("shutdown")

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
            RuntimeLifecycleStep("fail", RuntimeStepKind.STARTUP_HOOK, failing_step),
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

    with pytest.raises(RuntimeLifecycleError, match="fail"):
        await runtime.start(plan)

    assert runtime.state == RuntimeEngineState.FAILED
    assert runtime.background_tasks == []
    assert "failing_step" in events
    assert "shutdown" in events
    assert events.index("failing_step") < events.index("shutdown")
    assert created_tasks and created_tasks[0].cancelled()


@pytest.mark.asyncio
async def test_runtime_engine_stop_condition_prevents_startup() -> None:
    runtime = RuntimeEngine(_context())
    start_called = False

    def task_factory() -> asyncio.Task:
        nonlocal start_called
        start_called = True
        return asyncio.create_task(asyncio.sleep(60), name="should_not_start")

    plan = RuntimeLifecyclePlan(
        stop_conditions=(
            RuntimeStopCondition(
                name="manual_stop",
                is_met=lambda: True,
                reason="operator_requested",
            ),
        ),
        startup_steps=(
            RuntimeLifecycleStep("worker", RuntimeStepKind.BACKGROUND_TASK, task_factory),
        ),
    )

    with pytest.raises(RuntimeStopRequested, match="manual_stop"):
        await runtime.start(plan)

    assert runtime.state == RuntimeEngineState.FAILED
    assert start_called is False


@pytest.mark.asyncio
async def test_runtime_engine_stop_condition_error_is_lifecycle_failure() -> None:
    runtime = RuntimeEngine(_context())

    def broken_condition() -> bool:
        raise ValueError("predicate failed")

    plan = RuntimeLifecyclePlan(
        stop_conditions=(
            RuntimeStopCondition(
                name="broken_stop_condition",
                is_met=broken_condition,
                reason="predicate_failed",
            ),
        ),
    )

    with pytest.raises(RuntimeLifecycleError, match="broken_stop_condition") as exc_info:
        await runtime.start(plan)

    assert not isinstance(exc_info.value, RuntimeStopRequested)
    assert runtime.state == RuntimeEngineState.FAILED


@pytest.mark.asyncio
async def test_runtime_engine_shutdown_timeout_escalates_failure() -> None:
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=0.01)

    async def hanging_shutdown() -> None:
        await asyncio.sleep(60)

    plan = RuntimeLifecyclePlan(
        startup_steps=(),
        shutdown_steps=(
            RuntimeLifecycleStep(
                "hang",
                RuntimeStepKind.SHUTDOWN_HOOK,
                hanging_shutdown,
                timeout_seconds=0.01,
                register_task=False,
            ),
        ),
        shutdown_step_timeout_seconds=0.01,
    )

    await runtime.start(plan)
    await runtime.shutdown(plan)

    assert runtime.state == RuntimeEngineState.FAILED
    assert runtime.graceful_shutdown_failed is True
    assert runtime.health_check().healthy is False


@pytest.mark.asyncio
async def test_cancelled_start_cancels_timeout_wrapped_startup_hook() -> None:
    # A startup hook bounded by ``timeout_seconds`` runs in its own task. If the
    # caller is cancelled before the timeout fires, that task must be cancelled
    # and drained, not left running after the runtime moves to FAILED.
    runtime = RuntimeEngine(_context(), shutdown_timeout_seconds=5.0)
    hook_started = asyncio.Event()
    hook_cancelled = asyncio.Event()

    async def slow_hook() -> None:
        hook_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            hook_cancelled.set()
            raise

    plan = RuntimeLifecyclePlan(
        startup_steps=(
            RuntimeLifecycleStep(
                "slow_hook",
                RuntimeStepKind.STARTUP_HOOK,
                slow_hook,
                timeout_seconds=5.0,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=5.0,
    )

    start_task = asyncio.create_task(runtime.start(plan))
    try:
        await asyncio.wait_for(hook_started.wait(), timeout=1.0)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=1.0)

        await asyncio.wait_for(hook_cancelled.wait(), timeout=1.0)
        assert runtime.state == RuntimeEngineState.FAILED
        assert runtime.background_tasks == []
    finally:
        if not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
