"""Runtime lifecycle plan construction for the live TradingEngine.

Builds the declarative RuntimeLifecyclePlan (startup/shutdown steps, their
dependencies and stop conditions) that the RuntimeEngine executes. Extracted
from strategy.py following the engine's collaborator-function pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.features.live_trade.engines.runtime.models import (
    RuntimeDependency,
    RuntimeLifecyclePlan,
    RuntimeLifecycleStep,
    RuntimeStepKind,
    RuntimeStopCondition,
)
from gpt_trader.features.live_trade.lifecycle import EngineState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


def build_runtime_lifecycle_plan(engine: TradingEngine) -> RuntimeLifecyclePlan:
    shutdown_timeout_seconds = getattr(
        engine.context.config,
        "runtime_shutdown_timeout_seconds",
        5.0,
    )
    return RuntimeLifecyclePlan(
        dependencies=(
            RuntimeDependency("config", engine.context.config),
            RuntimeDependency("container", engine.context.container),
            RuntimeDependency("broker", engine.context.broker),
            RuntimeDependency("risk_manager", engine.context.risk_manager, required=False),
            RuntimeDependency("event_store", engine.context.event_store, required=False),
            RuntimeDependency("orders_store", engine.context.orders_store, required=False),
            RuntimeDependency(
                "notification_service",
                engine.context.notification_service,
                required=False,
            ),
        ),
        stop_conditions=(
            RuntimeStopCondition(
                name="engine_error_state",
                is_met=lambda: engine.state == EngineState.ERROR,
                reason="trading_engine_error",
            ),
        ),
        startup_steps=(
            RuntimeLifecycleStep(
                name="engine_state_starting",
                kind=RuntimeStepKind.STARTUP_HOOK,
                callback=engine._runtime_mark_starting,
            ),
            RuntimeLifecycleStep(
                name="price_history_rehydrate",
                kind=RuntimeStepKind.STARTUP_HOOK,
                callback=engine._runtime_rehydrate_once,
            ),
            RuntimeLifecycleStep(
                name="runtime_start_event",
                kind=RuntimeStepKind.STARTUP_HOOK,
                callback=engine._record_runtime_start,
            ),
            RuntimeLifecycleStep(
                name="engine_state_running",
                kind=RuntimeStepKind.STARTUP_HOOK,
                callback=engine._runtime_mark_running,
            ),
            RuntimeLifecycleStep(
                name="trading_loop",
                kind=RuntimeStepKind.BACKGROUND_TASK,
                callback=engine._start_trading_loop_task,
            ),
            RuntimeLifecycleStep(
                name="heartbeat_service",
                kind=RuntimeStepKind.HEARTBEAT,
                callback=engine._heartbeat.start,
            ),
            RuntimeLifecycleStep(
                name="status_reporter",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._status_reporter.start,
            ),
            RuntimeLifecycleStep(
                name="health_check_runner",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._health_check_runner.start,
            ),
            RuntimeLifecycleStep(
                name="system_maintenance_prune",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._system_maintenance.start_prune_loop,
            ),
            RuntimeLifecycleStep(
                name="runtime_guard_checkpoint",
                kind=RuntimeStepKind.POLICY_CHECKPOINT,
                callback=engine._runtime_guard_checkpoint,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="runtime_guard_sweep",
                kind=RuntimeStepKind.POLICY_CHECKPOINT,
                callback=engine._start_runtime_guard_task,
            ),
            RuntimeLifecycleStep(
                name="streaming",
                kind=RuntimeStepKind.STREAMING,
                callback=engine._runtime_start_streaming,
            ),
            RuntimeLifecycleStep(
                name="ws_health_watchdog",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._start_ws_health_watchdog_task,
            ),
        ),
        shutdown_steps=(
            RuntimeLifecycleStep(
                name="ws_health_watchdog",
                kind=RuntimeStepKind.SHUTDOWN_HOOK,
                callback=engine._stop_ws_health_watchdog_task,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="streaming",
                kind=RuntimeStepKind.STREAMING,
                callback=engine._runtime_stop_streaming,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="health_check_runner",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._health_check_runner.stop,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="system_maintenance",
                kind=RuntimeStepKind.SHUTDOWN_HOOK,
                callback=engine._system_maintenance.stop,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="status_reporter",
                kind=RuntimeStepKind.HEALTH,
                callback=engine._status_reporter.stop,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="heartbeat_service",
                kind=RuntimeStepKind.HEARTBEAT,
                callback=engine._heartbeat.stop,
                timeout_seconds=shutdown_timeout_seconds,
                register_task=False,
            ),
            RuntimeLifecycleStep(
                name="broker_call_executor",
                kind=RuntimeStepKind.SHUTDOWN_HOOK,
                callback=engine._runtime_shutdown_broker_calls,
                register_task=False,
            ),
        ),
        task_cleanup_timeout_seconds=shutdown_timeout_seconds,
        shutdown_step_timeout_seconds=shutdown_timeout_seconds,
    )
