"""Legacy execution coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinator_facades import (
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
)
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import (
    ExecutionCoordinator as _ExecutionCoordinator,
)
from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.utilities.async_utils import run_in_thread
from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.core.interfaces import Order
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = get_logger(__name__, component="execution_coordinator_facade")


class ExecutionCoordinator(
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
    _ExecutionCoordinator,
):
    """Compatibility layer preserving the historical ExecutionCoordinator API."""

    def __init__(self, bot: PerpsBot | None) -> None:
        context = self._setup_facade(
            bot,
            overrides={"execution_coordinator": self} if bot is not None else None,
            placeholder_factory=self._placeholder_context,
        )
        super().__init__(context)
        if bot is not None:
            self._sync_bot(context)

    # ------------------------------------------------------------------
    @ContextPreservingCoordinator.context_action(pass_context=True)
    def init_execution(self, context: CoordinatorContext) -> None:
        updated = self.initialize(context)
        self.update_context(updated)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    def ensure_order_lock(self) -> asyncio.Lock:  # type: ignore[override]
        return super().ensure_order_lock()

    @ContextPreservingCoordinator.context_action(sync_after=True)
    async def execute_decision(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        await super().execute_decision(*args, **kwargs)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    async def place_order(  # type: ignore[override]
        self,
        exec_engine: Any,
        **kwargs: Any,
    ) -> Order | None:
        return await super().place_order(exec_engine, **kwargs)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    async def place_order_inner(  # type: ignore[override]
        self,
        exec_engine: Any,
        **kwargs: Any,
    ) -> Order | None:
        return await super().place_order_inner(exec_engine, **kwargs)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    def _get_order_reconciler(self) -> OrderReconciler:  # type: ignore[override]
        return super()._get_order_reconciler()

    def reset_order_reconciler(self) -> None:  # type: ignore[override]
        super().reset_order_reconciler()

    @ContextPreservingCoordinator.context_action()
    async def run_runtime_guards(self) -> None:  # type: ignore[override]
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
                        logger.error(
                            "Error while running runtime guards",
                            operation="execution_runtime_guards",
                            stage="loop",
                            error=str(exc),
                            exc_info=True,
                        )
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    @ContextPreservingCoordinator.context_action()
    async def run_order_reconciliation(  # type: ignore[override]
        self,
        interval_seconds: int = 45,
    ) -> None:
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
                    logger.debug(
                        "Order reconciliation loop error",
                        operation="execution_order_reconcile",
                        stage="loop",
                        error=str(exc),
                        exc_info=True,
                    )
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise

    @ContextPreservingCoordinator.context_action(sync_after=True)
    async def _run_order_reconciliation_cycle(  # type: ignore[override]
        self,
        reconciler: OrderReconciler,
    ) -> None:
        await super()._run_order_reconciliation_cycle(reconciler)

    # ------------------------------------------------------------------
    def _placeholder_context(self) -> CoordinatorContext:
        placeholder_config = BotConfig(profile=Profile.PROD)
        return CoordinatorContext(
            config=placeholder_config,
            registry=ServiceRegistry(config=placeholder_config),
            symbols=(),
            bot_id="perps_bot",
        )

    def _context_overrides(self, bot: PerpsBot) -> dict[str, Any]:  # type: ignore[override]
        return {"execution_coordinator": self}

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
