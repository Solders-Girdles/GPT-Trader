"""Legacy telemetry coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.coordinator_facades import (
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
)
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.telemetry import (
    TelemetryCoordinator as _TelemetryCoordinator,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class TelemetryCoordinator(
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
    _TelemetryCoordinator,
):
    """Compatibility layer preserving the historical telemetry coordinator API."""

    def __init__(self, bot: PerpsBot) -> None:
        context = self._setup_facade(bot)
        super().__init__(context)
        self._sync_bot(context)

    # ------------------------------------------------------------------
    @ContextPreservingCoordinator.context_action(pass_context=True)
    def bootstrap(self, context: CoordinatorContext) -> None:
        updated = self.initialize(context)
        self.update_context(updated)

    def init_accounting_services(self) -> None:
        self.bootstrap()

    def init_market_services(self) -> None:
        pass

    @ContextPreservingCoordinator.context_action()
    def start_streaming_if_configured(self) -> None:
        if self._should_enable_streaming():
            self._schedule_coroutine(self._start_streaming())

    def start_streaming_background(self) -> None:
        self.start_streaming_if_configured()

    @ContextPreservingCoordinator.context_action()
    def stop_streaming_background(self) -> None:
        self._schedule_coroutine(self._stop_streaming())

    @ContextPreservingCoordinator.context_action()
    def restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        relevant = {"perps_enable_streaming", "perps_stream_level", "symbols"}
        if not relevant.intersection(diff.keys()):
            return

        should_stream = self._should_enable_streaming()

        async def _apply_restart() -> None:
            try:
                await self._stop_streaming()
            except Exception:
                logger.exception("Failed to stop existing streaming task during restart")
            if should_stream:
                try:
                    await self._start_streaming()
                except Exception:
                    logger.exception("Failed to start streaming after config change")

        self._schedule_coroutine(_apply_restart())

    @ContextPreservingCoordinator.context_action()
    async def run_account_telemetry(self, interval_seconds: int = 300) -> None:
        await self._run_account_telemetry(interval_seconds)

    @ContextPreservingCoordinator.context_action()
    def ensure_streaming_task(self) -> asyncio.Task[Any] | None:
        return self._stream_task

    # ------------------------------------------------------------------
    def _sync_bot(self, context: CoordinatorContext) -> None:
        bot = self._bot
        if hasattr(bot, "registry"):
            bot.registry = context.registry

        account_manager = context.registry.extras.get("account_manager")
        if account_manager is not None and hasattr(bot, "account_manager"):
            bot.account_manager = account_manager

        account_telemetry = context.registry.extras.get("account_telemetry")
        if account_telemetry is not None and hasattr(bot, "account_telemetry"):
            bot.account_telemetry = account_telemetry
            if hasattr(bot, "system_monitor") and callable(
                getattr(bot.system_monitor, "attach_account_telemetry", None)
            ):
                try:
                    bot.system_monitor.attach_account_telemetry(account_telemetry)
                except Exception:
                    logger.debug(
                        "Failed to attach account telemetry to system monitor", exc_info=True
                    )

        market_monitor = context.registry.extras.get("market_monitor")
        if market_monitor is not None and hasattr(bot, "market_monitor"):
            bot.market_monitor = market_monitor
            setattr(bot, "_market_monitor", market_monitor)

        intx_service = context.registry.extras.get("intx_portfolio_service")
        if intx_service is not None and hasattr(bot, "intx_portfolio_service"):
            bot.intx_portfolio_service = intx_service

    def _schedule_coroutine(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Execute a coordinator coroutine on the running loop or in a safe fallback."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(coro)
            return

        task_loop: asyncio.AbstractEventLoop | None = None
        if self._stream_task is not None:
            try:
                task_loop = self._stream_task.get_loop()
            except Exception:
                task_loop = None

        if task_loop and task_loop.is_running():
            task_loop.call_soon_threadsafe(asyncio.create_task, coro)
            return

        if loop is None:
            asyncio.run(coro)
        else:
            loop.run_until_complete(coro)


__all__ = ["TelemetryCoordinator"]
