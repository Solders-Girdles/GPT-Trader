from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.engines.base import CoordinatorContext

from .telemetry_services import run_account_telemetry as _run_account_telemetry_impl
logger = get_logger(__name__, component="telemetry_coordinator")


async def start_background_tasks(coordinator: "TelemetryCoordinator") -> list[asyncio.Task[Any]]:
    tasks: list[asyncio.Task[Any]] = []

    extras = getattr(coordinator.context.registry, "extras", {})
    account_telemetry = extras.get("account_telemetry") if isinstance(extras, dict) else None
    if account_telemetry and account_telemetry.supports_snapshots():
        interval = coordinator.context.config.account_telemetry_interval
        coro = _run_account_telemetry_impl(coordinator, interval)
        task = asyncio.create_task(coro)
        if not isinstance(task, asyncio.Task):  # pragma: no cover - testing fallback
            try:
                coro.close()
            except Exception:
                pass
        coordinator._register_background_task(task)
        tasks.append(task)
        logger.info(
            "Started account telemetry background task",
            interval=interval,
            operation="telemetry_tasks",
            stage="account_telemetry",
        )

    if coordinator._should_enable_streaming():
        try:
            stream_task = await coordinator._start_streaming()
            if stream_task is not None:
                coordinator._register_background_task(stream_task)
                tasks.append(stream_task)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to start streaming background task",
                error=str(exc),
                exc_info=True,
                operation="telemetry_tasks",
                stage="stream_start",
            )

    return tasks
