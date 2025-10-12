"""Lifecycle control utilities for :class:`bot_v2.orchestration.perps_bot.PerpsBot`."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = logging.getLogger(__name__)


class LifecycleManager:
    """Coordinate PerpsBot startup, run-loop execution, and shutdown."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        """Initialise orchestration collaborators required before running."""

        bot = self._bot
        bot.runtime_coordinator.bootstrap()
        bot.telemetry_coordinator.bootstrap()

    # ------------------------------------------------------------------
    async def run(self, single_cycle: bool = False) -> None:
        """Execute the PerpsBot runtime loop."""

        bot = self._bot
        logger.info("Starting Perps Bot - Profile: %s", bot.config.profile.value)
        bot.running = True
        background_tasks: list[asyncio.Task[Any]] = []
        try:
            if not bot.config.dry_run:
                await bot.runtime_coordinator.reconcile_state_on_startup()
                if not single_cycle:
                    background_tasks.append(
                        asyncio.create_task(bot.execution_coordinator.run_runtime_guards())
                    )
                    background_tasks.append(
                        asyncio.create_task(bot.execution_coordinator.run_order_reconciliation())
                    )
                    background_tasks.append(
                        asyncio.create_task(bot.system_monitor.run_position_reconciliation())
                    )
                    account_telemetry = getattr(bot, "account_telemetry", None)
                    if account_telemetry and account_telemetry.supports_snapshots():
                        background_tasks.append(
                            asyncio.create_task(
                                bot.telemetry_coordinator.run_account_telemetry(
                                    bot.config.account_telemetry_interval
                                )
                            )
                        )
            else:
                logger.info("Dry-run: skipping startup reconciliation and background guard loops")

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
            logger.info("Interrupted by user")
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive catch, behaviour verified indirectly
            logger.error("Bot error: %s", exc, exc_info=True)
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
        logger.info("Shutting down bot...")
        bot.running = False
        try:
            bot.telemetry_coordinator.stop_streaming_background()
        except Exception as exc:  # pragma: no cover - defensive shutdown logging
            logger.debug("Failed to stop WS thread cleanly: %s", exc, exc_info=True)
