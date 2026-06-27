"""Runtime lifecycle coordinator for live trading engines."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.engines.runtime.models import (
    RuntimeContextValidationError,
    RuntimeDependency,
    RuntimeEngineState,
    RuntimeLifecycleError,
    RuntimeLifecycleEvent,
    RuntimeLifecyclePlan,
    RuntimeLifecycleStep,
    RuntimeStepKind,
    RuntimeStopCondition,
    RuntimeStopRequested,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="runtime_engine")

TaskRegistrar = Callable[[asyncio.Task[Any]], None]
ResultT = TypeVar("ResultT")


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.debug(
            "Runtime lifecycle task completed after timeout",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation="runtime_lifecycle",
            stage="late_task_completion",
        )


def _coerce_timeout(value: float | int | None, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    return max(timeout, 0.0)


class RuntimeEngine(BaseEngine):
    """Owns runtime startup, shutdown, task cleanup, and escalation boundaries.

    The coordinator executes an explicit lifecycle plan supplied by the runtime
    owner. It deliberately does not decide strategy actions or submit orders.
    """

    def __init__(
        self,
        context: CoordinatorContext,
        *,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        super().__init__(context)
        self._state = RuntimeEngineState.IDLE
        self._events: list[RuntimeLifecycleEvent] = []
        self._named_tasks: dict[str, asyncio.Task[Any]] = {}
        self._shutdown_timeout_seconds = _coerce_timeout(shutdown_timeout_seconds, 5.0)
        self._last_error: str | None = None
        self._graceful_shutdown_failed = False
        self._drain_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_owner: asyncio.Task[Any] | None = None

    @property
    def name(self) -> str:
        return "runtime"

    @property
    def state(self) -> RuntimeEngineState:
        return self._state

    @property
    def events(self) -> tuple[RuntimeLifecycleEvent, ...]:
        return tuple(self._events)

    @property
    def graceful_shutdown_failed(self) -> bool:
        return self._graceful_shutdown_failed

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        selected_context = context or self.context
        self.update_context(selected_context)
        return selected_context

    async def start(
        self,
        plan: RuntimeLifecyclePlan,
        *,
        register_task: TaskRegistrar | None = None,
    ) -> list[asyncio.Task[Any]]:
        """Execute startup steps once and return the tracked background tasks."""
        return await self._run_lifecycle_serialized(
            lambda: self._start(plan, register_task=register_task)
        )

    async def _start(
        self,
        plan: RuntimeLifecyclePlan,
        *,
        register_task: TaskRegistrar | None,
    ) -> list[asyncio.Task[Any]]:
        if self._state == RuntimeEngineState.RUNNING:
            self._record_event("start_idempotent", "runtime", success=True)
            return self.background_tasks
        if self._state == RuntimeEngineState.DRAINING:
            raise RuntimeLifecycleError("Runtime startup requested while shutdown is draining")

        self._state = RuntimeEngineState.INITIALIZING
        self._last_error = None
        self._graceful_shutdown_failed = False
        self._record_event("state_initializing", "runtime", success=True)

        try:
            self.validate_context(plan.dependencies)
            self._check_stop_conditions(plan.stop_conditions)
            for step in plan.startup_steps:
                await self._run_step(step, register_task=register_task)
        except asyncio.CancelledError:
            self._last_error = "startup cancelled"
            logger.info(
                "Runtime startup cancelled; draining registered tasks",
                operation="runtime_lifecycle",
                stage="startup",
            )
            await self._run_drain(plan, failed=True, reason="startup_cancelled")
            raise
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception(
                "Runtime startup failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="runtime_lifecycle",
                stage="startup",
            )
            await self._run_drain(plan, failed=True, reason="startup_failed")
            if isinstance(exc, RuntimeLifecycleError):
                raise
            raise RuntimeLifecycleError(f"Runtime startup failed: {exc}") from exc

        self._state = RuntimeEngineState.RUNNING
        self._record_event("state_running", "runtime", success=True)
        return self.background_tasks

    async def shutdown(self, plan: RuntimeLifecyclePlan | None = None) -> None:
        """Run shutdown hooks and cancel tracked tasks within bounded timeouts."""
        await self._run_lifecycle_serialized(lambda: self._shutdown(plan))

    async def _shutdown(self, plan: RuntimeLifecyclePlan | None) -> None:
        if self._state in {RuntimeEngineState.TERMINATED, RuntimeEngineState.FAILED}:
            if self._tracked_tasks:
                if plan is None:
                    plan = RuntimeLifecyclePlan(
                        task_cleanup_timeout_seconds=self._shutdown_timeout_seconds,
                        shutdown_step_timeout_seconds=self._shutdown_timeout_seconds,
                    )
                await self._cleanup_tasks(plan.task_cleanup_timeout_seconds)
            self._record_event("shutdown_idempotent", "runtime", success=True)
            return
        if plan is None:
            plan = RuntimeLifecyclePlan(
                task_cleanup_timeout_seconds=self._shutdown_timeout_seconds,
                shutdown_step_timeout_seconds=self._shutdown_timeout_seconds,
            )
        await self._run_drain(plan, failed=False, reason="shutdown_called")

    @property
    def background_tasks(self) -> list[asyncio.Task[Any]]:
        return list(self._named_tasks.values())

    def validate_context(self, dependencies: Sequence[RuntimeDependency]) -> None:
        missing = [
            dependency.name
            for dependency in dependencies
            if dependency.required and dependency.value is None
        ]
        if missing:
            message = "Missing required runtime dependencies: " + ", ".join(sorted(missing))
            self._record_event("context_validation", RuntimeStepKind.STARTUP_HOOK, False, message)
            raise RuntimeContextValidationError(message)

        optional_missing = [
            dependency.name
            for dependency in dependencies
            if not dependency.required and dependency.value is None
        ]
        if optional_missing:
            logger.info(
                "Runtime optional dependencies unavailable",
                missing=optional_missing,
                operation="runtime_lifecycle",
                stage="context_validation",
            )
        self._record_event("context_validation", RuntimeStepKind.STARTUP_HOOK, True)

    def health_check(self) -> HealthStatus:
        healthy = self._state == RuntimeEngineState.RUNNING and not self._graceful_shutdown_failed
        return HealthStatus(
            healthy=healthy,
            component=self.name,
            details={
                "state": self._state.value,
                "tracked_tasks": len(self._named_tasks),
                "graceful_shutdown_failed": self._graceful_shutdown_failed,
            },
            error=self._last_error,
        )

    def _check_stop_conditions(self, stop_conditions: Sequence[RuntimeStopCondition]) -> None:
        active: list[str] = []
        for stop_condition in stop_conditions:
            try:
                if stop_condition.is_met():
                    active.append(f"{stop_condition.name}:{stop_condition.reason}")
            except Exception as exc:
                message = (
                    f"Runtime stop condition {stop_condition.name!r} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_event("stop_condition", RuntimeStepKind.STARTUP_HOOK, False, message)
                raise RuntimeLifecycleError(message) from exc

        if active:
            message = "Runtime startup stopped by condition(s): " + ", ".join(active)
            self._record_event("stop_condition", RuntimeStepKind.STARTUP_HOOK, False, message)
            raise RuntimeStopRequested(message)
        self._record_event("stop_condition", RuntimeStepKind.STARTUP_HOOK, True)

    async def _run_step(
        self,
        step: RuntimeLifecycleStep,
        *,
        register_task: TaskRegistrar | None = None,
    ) -> None:
        try:
            result = step.callback()
            if inspect.isawaitable(result) and not isinstance(result, asyncio.Task):
                if step.timeout_seconds is None:
                    result = await result
                else:
                    timeout = _coerce_timeout(step.timeout_seconds, self._shutdown_timeout_seconds)
                    result = await self._await_with_timeout(
                        result,
                        timeout=timeout,
                        task_name=step.name,
                    )
            self._register_step_tasks(step, result, register_task=register_task)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._record_event(step.name, step.kind, False, message)
            if step.required:
                raise RuntimeLifecycleError(f"Runtime step {step.name!r} failed: {exc}") from exc
            logger.warning(
                "Optional runtime lifecycle step failed",
                step=step.name,
                kind=step.kind.value,
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="runtime_lifecycle",
            )
            return
        self._record_event(step.name, step.kind, True)

    def _register_step_tasks(
        self,
        step: RuntimeLifecycleStep,
        result: Any,
        *,
        register_task: TaskRegistrar | None,
    ) -> None:
        if not step.register_task or result is None:
            return

        tasks = self._normalize_tasks(result)
        for index, task in enumerate(tasks):
            task_name = task.get_name() if hasattr(task, "get_name") else ""
            name = task_name or step.name
            if name in self._named_tasks and self._named_tasks[name] is not task:
                name = f"{step.name}_{index}"
            self._track_task(name, task)
            self._register_background_task(task)
            if register_task is not None:
                register_task(task)

    @property
    def _tracked_tasks(self) -> list[asyncio.Task[Any]]:
        self._forget_completed_tasks()
        return list(self._named_tasks.values())

    def _track_task(self, name: str, task: asyncio.Task[Any]) -> None:
        selected_name = name
        suffix = 1
        while selected_name in self._named_tasks and self._named_tasks[selected_name] is not task:
            suffix += 1
            selected_name = f"{name}_{suffix}"
        self._named_tasks[selected_name] = task

    def _forget_task(self, task: asyncio.Task[Any]) -> None:
        self._named_tasks = {
            name: tracked_task
            for name, tracked_task in self._named_tasks.items()
            if tracked_task is not task
        }
        self._background_tasks = [
            tracked_task for tracked_task in self._background_tasks if tracked_task is not task
        ]

    def _forget_completed_tasks(self) -> None:
        for task in list(self._named_tasks.values()):
            if task.done():
                _consume_task_result(task)
                self._forget_task(task)

    def _consume_and_forget_task(self, task: asyncio.Task[Any]) -> None:
        _consume_task_result(task)
        self._forget_task(task)

    def _normalize_tasks(self, result: Any) -> list[asyncio.Task[Any]]:
        if isinstance(result, asyncio.Task):
            return [result]
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            return [task for task in result if isinstance(task, asyncio.Task)]
        return []

    async def _run_lifecycle_serialized(
        self,
        operation: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        current_task = asyncio.current_task()
        if current_task is not None and (
            current_task is self._lifecycle_owner or current_task is self._drain_task
        ):
            return await operation()

        async with self._lifecycle_lock:
            self._lifecycle_owner = current_task
            try:
                return await operation()
            finally:
                if self._lifecycle_owner is current_task:
                    self._lifecycle_owner = None

    async def _run_drain(
        self,
        plan: RuntimeLifecyclePlan,
        *,
        failed: bool,
        reason: str,
    ) -> None:
        active_drain = self._drain_task
        if active_drain is not None and not active_drain.done():
            if active_drain is asyncio.current_task():
                return
            try:
                await asyncio.shield(active_drain)
            finally:
                if self._drain_task is active_drain and active_drain.done():
                    self._drain_task = None
            return

        drain_task = asyncio.create_task(
            self._drain(plan, failed=failed, reason=reason),
            name="runtime_drain",
        )
        self._drain_task = drain_task
        try:
            await asyncio.shield(drain_task)
        finally:
            if self._drain_task is drain_task and drain_task.done():
                self._drain_task = None

    async def _drain(
        self,
        plan: RuntimeLifecyclePlan,
        *,
        failed: bool,
        reason: str,
    ) -> None:
        if self._state == RuntimeEngineState.DRAINING:
            return

        self._state = RuntimeEngineState.DRAINING
        self._record_event("state_draining", "runtime", success=True)
        for step in plan.shutdown_steps:
            await self._run_shutdown_step(step, plan)

        await self._cleanup_tasks(plan.task_cleanup_timeout_seconds)
        if failed or self._graceful_shutdown_failed:
            self._state = RuntimeEngineState.FAILED
            self._record_event(reason, "runtime", success=False, error=self._last_error)
        else:
            self._state = RuntimeEngineState.TERMINATED
            self._record_event("state_terminated", "runtime", success=True)

    async def _run_shutdown_step(
        self,
        step: RuntimeLifecycleStep,
        plan: RuntimeLifecyclePlan,
    ) -> None:
        timeout = step.timeout_seconds
        if timeout is None:
            timeout = plan.shutdown_step_timeout_seconds
        timeout = _coerce_timeout(timeout, self._shutdown_timeout_seconds)

        try:
            result = step.callback()
            if inspect.isawaitable(result):
                await self._await_with_timeout(
                    result,
                    timeout=timeout,
                    task_name=step.name,
                )
        except Exception as exc:
            self._graceful_shutdown_failed = True
            self._last_error = str(exc)
            message = f"{type(exc).__name__}: {exc}"
            self._record_event(step.name, step.kind, False, message)
            logger.warning(
                "Runtime shutdown step failed",
                step=step.name,
                kind=step.kind.value,
                error_type=type(exc).__name__,
                error_message=str(exc),
                timeout_seconds=timeout,
                operation="runtime_lifecycle",
                stage="shutdown",
            )
            return
        self._record_event(step.name, step.kind, True)

    async def _await_with_timeout(
        self,
        awaitable: Awaitable[Any],
        *,
        timeout: float,
        task_name: str,
    ) -> Any:
        task = asyncio.ensure_future(awaitable)
        done, pending = await asyncio.wait({task}, timeout=timeout)
        if pending:
            self._track_task(f"{task_name}_timeout", task)
            task.cancel()
            task.add_done_callback(self._consume_and_forget_task)
            raise TimeoutError(f"timed out after {timeout:.3f}s")
        return await next(iter(done))

    async def _cleanup_tasks(self, timeout_seconds: float) -> None:
        tasks = self._tracked_tasks
        for task in tasks:
            task.cancel()

        effective_timeout = _coerce_timeout(timeout_seconds, self._shutdown_timeout_seconds)
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=effective_timeout)
            for task in done:
                _consume_task_result(task)
            if pending:
                for task in pending:
                    task.cancel()
                    task.add_done_callback(self._consume_and_forget_task)
                self._graceful_shutdown_failed = True
                self._last_error = f"timed out waiting for {len(pending)} runtime task(s)"
                logger.warning(
                    "Runtime task cleanup timed out",
                    task_count=len(tasks),
                    pending_task_count=len(pending),
                    timeout_seconds=effective_timeout,
                    operation="runtime_lifecycle",
                    stage="task_cleanup",
                )
                self._record_event(
                    "task_cleanup",
                    RuntimeStepKind.SHUTDOWN_HOOK,
                    False,
                    "timeout",
                )
            else:
                for task in done:
                    self._forget_task(task)
                self._record_event("task_cleanup", RuntimeStepKind.SHUTDOWN_HOOK, True)
        else:
            self._record_event("task_cleanup", RuntimeStepKind.SHUTDOWN_HOOK, True)

        self._forget_completed_tasks()

    def _record_event(
        self,
        name: str,
        kind: RuntimeStepKind | str,
        success: bool,
        error: str | None = None,
    ) -> None:
        self._events.append(
            RuntimeLifecycleEvent(
                name=name,
                kind=kind,
                state=self._state,
                success=success,
                error=error,
            )
        )


__all__ = ["RuntimeEngine"]
