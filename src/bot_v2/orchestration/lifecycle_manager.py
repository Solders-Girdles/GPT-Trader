"""Lifecycle control utilities for :class:`bot_v2.orchestration.perps_bot.PerpsBot`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = get_logger(__name__, component="lifecycle_manager")


class LifecycleManager:
    """Coordinate PerpsBot startup, run-loop execution, and shutdown."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        """Initialise orchestration collaborators required before running."""

        bot = self._bot
        logger.info(
            "Initializing coordinators",
            operation="lifecycle_bootstrap",
            stage="start",
            profile=bot.config.profile.value,
        )
        updated_context = bot._coordinator_registry.initialize_all()
        bot._coordinator_context = updated_context

        if updated_context.registry is not bot.registry:
            bot.registry = updated_context.registry
        if updated_context.broker is not None:
            try:
                bot.broker = updated_context.broker
            except Exception:
                logger.debug(
                    "Failed to set broker during bootstrap",
                    operation="lifecycle_bootstrap",
                    stage="broker_assignment",
                    exc_info=True,
                )
        if updated_context.risk_manager is not None:
            try:
                bot.risk_manager = updated_context.risk_manager
            except Exception:
                logger.debug(
                    "Failed to set risk manager during bootstrap",
                    operation="lifecycle_bootstrap",
                    stage="risk_assignment",
                    exc_info=True,
                )

        extras = updated_context.registry.extras
        account_manager = extras.get("account_manager")
        if account_manager is not None and hasattr(bot, "account_manager"):
            bot.account_manager = account_manager
        account_telemetry = extras.get("account_telemetry")
        if account_telemetry is not None and hasattr(bot, "account_telemetry"):
            bot.account_telemetry = account_telemetry
        market_monitor = extras.get("market_monitor")
        if market_monitor is not None and hasattr(bot, "market_monitor"):
            bot.market_monitor = market_monitor
            setattr(bot, "_market_monitor", market_monitor)

        try:
            bot.strategy_orchestrator.init_strategy()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to initialise strategy orchestrator during bootstrap",
                error=str(exc),
                operation="lifecycle_bootstrap",
                stage="strategy_init",
                exc_info=True,
            )

        logger.info(
            "Coordinator initialization complete",
            operation="lifecycle_bootstrap",
            stage="complete",
            symbols=list(bot.symbols),
        )

    # ------------------------------------------------------------------
    async def run(self, single_cycle: bool = False) -> None:
        """Execute the PerpsBot runtime loop."""

        bot = self._bot
        logger.info(
            "Starting Perps Bot run loop",
            operation="lifecycle_run",
            stage="start",
            profile=bot.config.profile.value,
            single_cycle=single_cycle,
            dry_run=bool(bot.config.dry_run),
        )
        bot.running = True
        background_tasks: list[asyncio.Task[Any]] = []
        try:
            if not bot.config.dry_run:
                await bot.runtime_coordinator.reconcile_state_on_startup()

            if not single_cycle and not bot.config.dry_run:
                logger.info(
                    "Starting coordinator background tasks",
                    operation="lifecycle_run",
                    stage="background_tasks_start",
                )
                coordinator_tasks = await bot._coordinator_registry.start_all_background_tasks()
                background_tasks.extend(coordinator_tasks)
                logger.info(
                    "Coordinator background tasks started",
                    operation="lifecycle_run",
                    stage="background_tasks_started",
                    task_count=len(coordinator_tasks),
                )

            if not bot.config.dry_run:
                background_tasks.append(
                    asyncio.create_task(bot.system_monitor.run_position_reconciliation())
                )

            await bot.run_cycle()
            bot.system_monitor.write_health_status(ok=True)
            bot.system_monitor.check_config_updates()

            if not single_cycle and not bot.config.dry_run:
                while bot.running:
                    await asyncio.sleep(bot.config.update_interval)
                    if not bot.running:
                        break
                    await bot.run_cycle()
                    bot.system_monitor.write_health_status(ok=True)
                    bot.system_monitor.check_config_updates()
        except KeyboardInterrupt:
            logger.info(
                "Run loop interrupted by user",
                operation="lifecycle_run",
                stage="interrupt",
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive catch, behaviour verified indirectly
            logger.error(
                "Run loop encountered error",
                operation="lifecycle_run",
                stage="exception",
                error=str(exc),
                exc_info=True,
            )
            bot.system_monitor.write_health_status(ok=False, error=str(exc))
        finally:
            bot.running = False
            for task in background_tasks:
                if not task.done():
                    task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            await self.shutdown()

    async def shutdown(self) -> None:
        """Tear down runtime resources gracefully."""

        bot = self._bot
        logger.info(
            "Shutting down bot",
            operation="lifecycle_shutdown",
            stage="start",
        )
        bot.running = False
        await bot._coordinator_registry.shutdown_all()
