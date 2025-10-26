from __future__ import annotations

import asyncio
import threading
from typing import Any, Tuple

from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.utilities.logging_patterns import get_logger

from .base import BaseCoordinator, CoordinatorContext
from .telemetry_background import start_background_tasks as _start_background_tasks
from .telemetry_health import (
    extract_mark_from_message as _extract_mark_from_message_impl,
    health_check as _health_check_impl,
    update_mark_and_metrics as _update_mark_and_metrics_impl,
)
from .telemetry_services import (
    NullAccountTelemetry,
    ensure_account_telemetry_stub,
    init_market_services as _init_market_services_impl,
    initialize_services as _initialize_services_impl,
    run_account_telemetry as _run_account_telemetry_impl,
)
from .telemetry_streaming import (
    _handle_stream_task_completion as _handle_stream_task_completion_impl,
    _run_stream_loop as _run_stream_loop_impl,
    _run_stream_loop_async as _run_stream_loop_async_impl,
    _schedule_coroutine as _schedule_coroutine_impl,
    _should_enable_streaming as _should_enable_streaming_impl,
    _start_streaming as _start_streaming_impl,
    _stop_streaming as _stop_streaming_impl,
    restart_streaming_if_needed as _restart_streaming_if_needed_impl,
    start_streaming_background as _start_streaming_background_impl,
    stop_streaming_background as _stop_streaming_background_impl,
)

logger = get_logger(__name__, component="telemetry_coordinator")


def _resolve_initialize_target(
    target: Any, context: CoordinatorContext | None
) -> Tuple["TelemetryCoordinator", CoordinatorContext]:
    if isinstance(target, type):
        if context is None:
            raise ValueError("Context must be provided when calling initialize on the class")
        instance = target(context)
        return instance, context
    if isinstance(target, CoordinatorContext):
        instance = TelemetryCoordinator(target)
        return instance, target
    coordinator: TelemetryCoordinator = target  # type: ignore[assignment]
    return coordinator, context or coordinator.context


class TelemetryCoordinator(BaseCoordinator):
    """Manages account telemetry services and market streaming for the bot."""

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._stream_task: asyncio.Task[Any] | None = None
        self._ws_stop: threading.Event | None = None
        self._market_monitor: MarketActivityMonitor | None = None
        self._pending_stream_config: tuple[list[str], int] | None = None
        self._loop_task_handle: asyncio.Task[Any] | None = None
        ensure_account_telemetry_stub(self)

    @property
    def name(self) -> str:
        return "telemetry"

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext | None:
        if isinstance(self, type):
            if context is None:
                return None
            instance = self(context)
            return instance.initialize(context)
        if isinstance(self, CoordinatorContext):
            instance = TelemetryCoordinator(self)
            return instance.initialize(self)

        coordinator: TelemetryCoordinator = self  # type: ignore[assignment]
        if context is None and not hasattr(coordinator, "context"):
            return None
        ctx = context or coordinator.context
        return _initialize_services_impl(coordinator, ctx)

    def init_market_services(self) -> None:
        updated = self.initialize(self.context)
        if isinstance(updated, CoordinatorContext):
            self.update_context(updated)

    start_streaming_background = _start_streaming_background_impl
    stop_streaming_background = _stop_streaming_background_impl
    restart_streaming_if_needed = _restart_streaming_if_needed_impl

    async def run_account_telemetry(self, interval_seconds: int = 300) -> None:
        await _run_account_telemetry_impl(self, interval_seconds)

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        return await _start_background_tasks(self)

    async def shutdown(self) -> None:
        logger.info("Shutting down telemetry coordinator...")
        await self._stop_streaming()
        await super().shutdown()

    health_check = _health_check_impl

    _schedule_coroutine = _schedule_coroutine_impl
    _start_streaming = _start_streaming_impl
    _stop_streaming = _stop_streaming_impl
    _handle_stream_task_completion = _handle_stream_task_completion_impl
    _run_stream_loop_async = _run_stream_loop_async_impl
    _run_stream_loop = _run_stream_loop_impl
    _should_enable_streaming = _should_enable_streaming_impl
    _extract_mark_from_message = staticmethod(_extract_mark_from_message_impl)
    _update_mark_and_metrics = _update_mark_and_metrics_impl


__all__ = ["TelemetryCoordinator"]
