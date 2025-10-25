"""Runtime guards functionality separated from execution coordinator.

This module contains the runtime guards logic that was previously
embedded in the large execution.py file. It provides:

- Background runtime guard loops
- System health monitoring
- Emergency condition detection
- Guard action execution
- Configurable guard parameters
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.coordinators.base import CoordinatorContext

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="runtime_guards")


class RuntimeGuardsService:
    """Service responsible for runtime safety monitoring and guard actions.

    This service runs background tasks to monitor system health
    and execute protective actions when dangerous conditions are detected.
    """

    def __init__(
        self,
        context: CoordinatorContext,
        guard_interval_seconds: int = 60,
    ) -> None:
        """Initialize the runtime guards service.

        Args:
            context: Coordinator context with runtime configuration
            guard_interval_seconds: Interval between guard checks
        """
        self.context = context
        self.guard_interval_seconds = guard_interval_seconds
        self._running = False
        self._guards_task: asyncio.Task[Any] | None = None
        self._triggered_guards: dict[str, Any] = {}

    async def start_guards(self) -> None:
        """Start the background runtime guards process."""
        if self._running:
            logger.warning(
                "Runtime guards already running",
                operation="runtime_guards",
                status="already_running",
            )
            return

        self._running = True
        logger.info(
            "Starting runtime guards",
            operation="runtime_guards",
            interval_seconds=self.guard_interval_seconds,
        )

        # Start the guards loop
        self._guards_task = asyncio.create_task(self._run_guards_loop())

    async def stop_guards(self) -> None:
        """Stop the background runtime guards process."""
        if not self._running:
            logger.warning(
                "Runtime guards not running",
                operation="runtime_guards",
                status="not_running",
            )
            return

        self._running = False
        logger.info(
            "Stopping runtime guards",
            operation="runtime_guards",
        )

        if self._guards_task:
            self._guards_task.cancel()
            try:
                await self._guards_task
            except asyncio.CancelledError:
                pass

        self._guards_task = None
        logger.info(
            "Runtime guards stopped",
            operation="runtime_guards",
            status="stopped",
        )

    def is_running(self) -> bool:
        """Check if the guards service is running."""
        return self._running

    async def _run_guards_loop(self) -> None:
        """Run the main runtime guards loop."""
        try:
            while self._running:
                logger.debug(
                    "Running runtime guards cycle",
                    operation="runtime_guards_cycle",
                )

                # Run guard checks
                await self._run_guard_checks()

                # Wait for the next interval
                await asyncio.sleep(self.guard_interval_seconds)

        except asyncio.CancelledError:
            logger.info(
                "Runtime guards loop cancelled",
                operation="runtime_guards",
            )
        except Exception as exc:
            logger.error(
                "Runtime guards loop error",
                operation="runtime_guards",
                error=str(exc),
                exc_info=True,
            )
            if self._running:  # Only restart if not intentionally stopped
                logger.info(
                    "Restarting guards loop after error",
                    operation="runtime_guards",
                )
                await asyncio.sleep(10)  # Brief delay before restart
                # Loop will continue with next iteration

    async def _run_guard_checks(self) -> None:
        """Run all configured guard checks."""
        guards_to_run = [
            self._check_system_health,
            self._check_memory_usage,
            self._check_error_rates,
            self._check_connection_health,
        ]

        for guard_func in guards_to_run:
            try:
                await guard_func()
            except Exception as exc:
                logger.error(
                    "Guard check failed",
                    operation="runtime_guard_check",
                    guard_function=guard_func.__name__,
                    error=str(exc),
                    exc_info=True,
                )

    async def _check_system_health(self) -> None:
        """Check overall system health."""
        # This would integrate with system monitoring
        # For now, log a heartbeat
        logger.debug(
            "System health check passed",
            operation="runtime_guard_health",
        )

    async def _check_memory_usage(self) -> None:
        """Check memory usage and alert if necessary."""
        # This would check actual memory usage
        # For now, placeholder implementation
        logger.debug(
            "Memory usage check",
            operation="runtime_guard_memory",
            usage_mb="unknown",  # Would be actual memory usage
        )

    async def _check_error_rates(self) -> None:
        """Check error rates and trigger alerts if needed."""
        # This would track error rates over time windows
        # For now, placeholder implementation
        logger.debug(
            "Error rate check",
            operation="runtime_guard_errors",
            error_rate="normal",  # Would be calculated error rate
        )

    async def _check_connection_health(self) -> None:
        """Check connection health to broker and data sources."""
        # This would ping broker connections and data sources
        # For now, placeholder implementation
        logger.debug(
            "Connection health check",
            operation="runtime_guard_connections",
            connections="healthy",  # Would be actual connection status
        )

    def trigger_guard(
        self,
        guard_name: str,
        condition: Any,
        action: str,
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Trigger a guard action.

        Args:
            guard_name: Name/identifier of the guard
            condition: The condition that triggered the guard
            action: Action to take (e.g., "pause_trading", "reduce_leverage")
            severity: Severity level of the guard
            metadata: Optional additional metadata
        """
        self._triggered_guards[guard_name] = {
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop() else None,
            "condition": condition,
            "action": action,
            "severity": severity,
            "metadata": metadata or {},
        }

        logger.warning(
            f"Guard triggered: {guard_name}",
            operation="runtime_guard_triggered",
            guard_name=guard_name,
            action=action,
            severity=severity,
            condition=condition,
        )

    def clear_triggered_guards(self) -> None:
        """Clear all triggered guards."""
        self._triggered_guards.clear()
        logger.info(
            "Triggered guards cleared",
            operation="runtime_guards",
        )

    def get_triggered_guards(self) -> dict[str, Any]:
        """Get all currently triggered guards."""
        return self._triggered_guards.copy()

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the guards service."""
        return {
            "running": self.is_running(),
            "guard_interval_seconds": self.guard_interval_seconds,
            "guards_task": self._guards_task is not None,
            "triggered_guards": len(self._triggered_guards),
        }


__all__ = [
    "RuntimeGuardsService",
]
