"""Legacy execution coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.context_builder import build_coordinator_context
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import (
    ExecutionCoordinator as _ExecutionCoordinator,
)
from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.utilities.async_utils import run_in_thread

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.core.interfaces import Order
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class ExecutionCoordinator(_ExecutionCoordinator):
    """Compatibility layer preserving the historical ExecutionCoordinator API."""

    def __init__(self, bot: PerpsBot | None) -> None:
        self._bot = bot
        if bot is None:
            placeholder_config = BotConfig(profile=Profile.PROD)
            context = CoordinatorContext(
                config=placeholder_config,
                registry=ServiceRegistry(config=placeholder_config),
                symbols=(),
                bot_id="perps_bot",
            )
            super().__init__(context)
            return
        context = self._build_context(bot)
        super().__init__(context)
        self._sync_bot(context)

    # ------------------------------------------------------------------
    def init_execution(self) -> None:
        context = self._refresh_context_from_bot()
        updated = self.initialize(context)
        self.update_context(updated)
        self._sync_bot(updated)

    def ensure_order_lock(self) -> asyncio.Lock:  # type: ignore[override]
        self._refresh_context_from_bot()
        lock = super().ensure_order_lock()
        self._sync_bot(self.context)
        return lock

    async def execute_decision(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._refresh_context_from_bot()
        await super().execute_decision(*args, **kwargs)
        self._sync_bot(self.context)

    async def place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:  # type: ignore[override]
        self._refresh_context_from_bot()
        result = await super().place_order(exec_engine, **kwargs)
        self._sync_bot(self.context)
        return result

    async def place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:  # type: ignore[override]
        self._refresh_context_from_bot()
        result = await super().place_order_inner(exec_engine, **kwargs)
        self._sync_bot(self.context)
        return result

    def _get_order_reconciler(self) -> OrderReconciler:  # type: ignore[override]
        self._refresh_context_from_bot()
        reconciler = super()._get_order_reconciler()
        self._sync_bot(self.context)
        return reconciler

    def reset_order_reconciler(self) -> None:  # type: ignore[override]
        super().reset_order_reconciler()

    async def run_runtime_guards(self) -> None:  # type: ignore[override]
        self._refresh_context_from_bot()
        await super().run_runtime_guards()

    async def _run_runtime_guards_loop(self) -> None:  # type: ignore[override]
        bot = self._bot
        if bot is None:
            await super()._run_runtime_guards_loop()
            return
        try:
            while getattr(bot, "running", True):
                self._refresh_context_from_bot()
                exec_engine = getattr(self.context.runtime_state, "exec_engine", None)
                if exec_engine is not None:
                    try:
                        await run_in_thread(exec_engine.run_runtime_guards)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.error("Error in runtime guards: %s", exc, exc_info=True)
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    async def run_order_reconciliation(self, interval_seconds: int = 45) -> None:  # type: ignore[override]
        self._refresh_context_from_bot()
        await super().run_order_reconciliation(interval_seconds=interval_seconds)

    async def _run_order_reconciliation_loop(self, interval_seconds: int = 45) -> None:  # type: ignore[override]
        bot = self._bot
        if bot is None:
            await super()._run_order_reconciliation_loop(interval_seconds=interval_seconds)
            return
        try:
            while getattr(bot, "running", True):
                self._refresh_context_from_bot()
                try:
                    reconciler = self._get_order_reconciler()
                    await super()._run_order_reconciliation_cycle(reconciler)
                    self._sync_bot(self.context)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Order reconciliation error: %s", exc, exc_info=True)
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise

    async def _run_order_reconciliation_cycle(self, reconciler: OrderReconciler) -> None:  # type: ignore[override]
        await super()._run_order_reconciliation_cycle(reconciler)
        self._sync_bot(self.context)

    # ------------------------------------------------------------------
    def _refresh_context_from_bot(self) -> CoordinatorContext:
        if self._bot is None:
            return self.context
        updated = self._build_context(self._bot)
        super().update_context(updated)
        return self.context

    def _build_context(self, bot: PerpsBot) -> CoordinatorContext:
        return build_coordinator_context(
            bot,
            overrides={
                "execution_coordinator": self,
            },
        )

    def _sync_bot(self, context: CoordinatorContext) -> None:
        bot = self._bot
        if bot is None:
            return
        bot.registry = context.registry
        if context.event_store is not None:
            bot.event_store = context.event_store
        if context.orders_store is not None:
            bot.orders_store = context.orders_store
        if context.product_cache is not None:
            if hasattr(bot, "_state") and hasattr(bot._state, "product_map"):
                bot._state.product_map = context.product_cache
        if context.broker is not None:
            bot.registry = bot.registry.with_updates(broker=context.broker)
        if context.risk_manager is not None:
            bot.registry = bot.registry.with_updates(risk_manager=context.risk_manager)


__all__ = ["ExecutionCoordinator"]
