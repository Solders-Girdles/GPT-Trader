"""Lifecycle management service for trading bot orchestration.

This service extracts the trading loop, background task spawning, and error handling
from PerpsBot, making time-based behavior mockable and testable without spinning up
the full bot.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class CycleRunner(Protocol):
    """Protocol for objects that can run a trading cycle."""

    async def run_cycle(self) -> None:
        """Execute a single trading cycle."""
        ...

    @property
    def running(self) -> bool:
        """Check if the bot is currently running."""
        ...

    @running.setter
    def running(self, value: bool) -> None:
        """Set the running state."""
        ...


class BackgroundTaskRegistry:
    """Registry for managing background tasks with lifecycle."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[Any]] = []
        self._factory_functions: list[Callable[[], asyncio.Task[Any]]] = []

    def register(self, task_factory: Callable[[], asyncio.Task[Any]]) -> None:
        """Register a task factory function.

        Args:
            task_factory: Function that creates and returns an asyncio.Task
        """
        self._factory_functions.append(task_factory)

    def spawn_all(self) -> list[asyncio.Task[Any]]:
        """Spawn all registered tasks.

        Returns:
            List of spawned tasks
        """
        self._tasks = [factory() for factory in self._factory_functions]
        return self._tasks

    async def cancel_all(self) -> None:
        """Cancel all running tasks and wait for cleanup."""
        if not self._tasks:
            return

        for task in self._tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def clear(self) -> None:
        """Clear all registered task factories."""
        self._factory_functions.clear()


class LifecycleService:
    """Manages the trading bot lifecycle including loops, tasks, and error handling.

    This service isolates the main trading loop, background task spawning, and
    error handling from the PerpsBot class, making time-based behavior mockable
    and testable.

    Example:
        >>> lifecycle = LifecycleService(bot)
        >>> lifecycle.register_background_task(
        ...     lambda: asyncio.create_task(bot.execution_coordinator.run_runtime_guards())
        ... )
        >>> await lifecycle.run(single_cycle=False)
    """

    def __init__(
        self,
        bot: PerpsBot,
        sleep_fn: Callable[[float], Any] | None = None,
    ) -> None:
        """Initialize the lifecycle service.

        Args:
            bot: The PerpsBot instance to manage
            sleep_fn: Optional custom sleep function for testing (defaults to asyncio.sleep)
        """
        self._bot = bot
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._task_registry = BackgroundTaskRegistry()

    def register_background_task(self, task_factory: Callable[[], asyncio.Task[Any]]) -> None:
        """Register a background task to run during the bot lifecycle.

        Args:
            task_factory: Function that creates and returns an asyncio.Task

        Example:
            >>> lifecycle.register_background_task(
            ...     lambda: asyncio.create_task(bot.execution_coordinator.run_runtime_guards())
            ... )
        """
        self._task_registry.register(task_factory)

    def configure_background_tasks(self, single_cycle: bool) -> None:
        """Configure background tasks based on bot configuration.

        This sets up the standard background tasks:
        - Runtime guards
        - Order reconciliation
        - Position reconciliation
        - Account telemetry (if supported)

        Args:
            single_cycle: If True, skip background tasks
        """
        if self._bot.config.dry_run or single_cycle:
            return

        # Runtime guards
        self._task_registry.register(
            lambda: asyncio.create_task(self._bot.execution_coordinator.run_runtime_guards())
        )

        # Order reconciliation
        self._task_registry.register(
            lambda: asyncio.create_task(self._bot.execution_coordinator.run_order_reconciliation())
        )

        # Position reconciliation
        self._task_registry.register(
            lambda: asyncio.create_task(self._bot.system_monitor.run_position_reconciliation())
        )

        # Account telemetry (if supported)
        if self._bot.account_telemetry.supports_snapshots():
            interval = self._bot.config.account_telemetry_interval

            self._task_registry.register(
                lambda: asyncio.create_task(self._run_account_telemetry(interval))
            )

        # Execution metrics export (every 60s)
        self._task_registry.register(
            lambda: asyncio.create_task(self._run_execution_metrics_export())
        )

    async def _run_account_telemetry(self, interval_seconds: int = 300) -> None:
        """Run account telemetry background task.

        Args:
            interval_seconds: Interval between telemetry snapshots
        """
        if not self._bot.account_telemetry.supports_snapshots():
            return
        await self._bot.account_telemetry.run(interval_seconds)

    async def _run_execution_metrics_export(self) -> None:
        """Run execution metrics export background task.

        Delegates to PerpsBot's _run_execution_metrics_export method.
        """
        await self._bot._run_execution_metrics_export(interval_seconds=60)

    async def run(self, single_cycle: bool = False) -> None:
        """Run the trading bot lifecycle.

        This is the main entry point for the bot's execution loop. It:
        1. Performs startup reconciliation (if not dry_run)
        2. Spawns background tasks (if not single_cycle)
        3. Runs the main trading loop
        4. Handles errors and cleanup

        Args:
            single_cycle: If True, run only one cycle and exit

        Raises:
            KeyboardInterrupt: Re-raised after cleanup
            Exception: Logged and written to health status
        """
        logger.info("Starting Perps Bot - Profile: %s", self._bot.config.profile.value)
        self._bot.running = True

        # Start metrics server
        if not self._bot.metrics_server.is_running:
            try:
                self._bot.metrics_server.start()
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")

        try:
            # Startup reconciliation
            if not self._bot.config.dry_run:
                await self._bot.runtime_coordinator.reconcile_state_on_startup()
                self._task_registry.spawn_all()
            else:
                logger.info("Dry-run: skipping startup reconciliation and background guard loops")

            # Main trading loop
            await self._run_trading_loop(single_cycle)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            raise

        except Exception as exc:
            logger.error("Bot error: %s", exc, exc_info=True)
            self._bot.system_monitor.write_health_status(ok=False, error=str(exc))
            raise

        finally:
            await self._cleanup()

    async def _run_trading_loop(self, single_cycle: bool) -> None:
        """Run the main trading loop.

        Args:
            single_cycle: If True, run only one cycle
        """
        # First cycle
        await self._run_single_cycle()

        # Continue looping if not single_cycle and not dry_run
        if not single_cycle and not self._bot.config.dry_run:
            while self._bot.running:
                await self._sleep_fn(self._bot.config.update_interval)

                if not self._bot.running:
                    break

                await self._run_single_cycle()

    async def _run_single_cycle(self) -> None:
        """Execute a single trading cycle and post-cycle tasks."""
        import time

        # Guard rail cycle checks (e.g., halt states)
        guardrails = getattr(self._bot, "guardrails", None)
        if guardrails is not None:
            guardrails.check_cycle({"profile": self._bot.config.profile.value})

        start_time = time.time()
        await self._bot.run_cycle()
        duration = time.time() - start_time

        # Record metrics
        profile = self._bot.config.profile.value
        if guardrails is not None:
            if self._bot.metrics_server:
                try:
                    daily_loss_value = float(guardrails.get_daily_loss())
                except (TypeError, ValueError):
                    daily_loss_value = 0.0

                try:
                    streak_value = int(guardrails.get_error_streak())
                except (TypeError, ValueError):
                    streak_value = 0

                self._bot.metrics_server.daily_loss_gauge.labels(profile=profile).set(
                    daily_loss_value
                )
                self._bot.metrics_server.update_error_streak(streak_value, profile=profile)
        self._bot.metrics_server.record_cycle_duration(duration, profile=profile)
        self._bot.metrics_server.update_uptime(profile=profile)

        self._bot.system_monitor.write_health_status(ok=True)
        self._bot.system_monitor.check_config_updates()

    async def _cleanup(self) -> None:
        """Clean up resources and shutdown the bot."""
        self._bot.running = False
        await self._task_registry.cancel_all()
        await self._bot.shutdown()
