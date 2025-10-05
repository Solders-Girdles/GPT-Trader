from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    Product,
)
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.orchestration.execution.engine_factory import ExecutionEngineFactory
from bot_v2.orchestration.execution.order_placement import OrderPlacementService
from bot_v2.orchestration.execution.runtime_supervisor import ExecutionRuntimeSupervisor
from bot_v2.orchestration.order_reconciler import OrderReconciler

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates execution engine interactions and order placement."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        self._order_reconciler: OrderReconciler | None = None
        self._order_placement_service: OrderPlacementService | None = None
        self._runtime_supervisor: ExecutionRuntimeSupervisor | None = None

    def init_execution(self) -> None:
        """Initialize execution engine using ExecutionEngineFactory."""
        bot = self._bot

        # Use factory to create engine
        bot.exec_engine = ExecutionEngineFactory.create_engine(
            broker=bot.broker,
            risk_manager=bot.risk_manager,
            event_store=bot.event_store,
            bot_id="perps_bot",
            enable_preview=bot.config.enable_order_preview,
        )

        # Register engine in registry
        extras = dict(bot.registry.extras)
        extras["execution_engine"] = bot.exec_engine
        bot.registry = bot.registry.with_updates(extras=extras)

    def _get_order_placement_service(self) -> OrderPlacementService:
        """Get or create OrderPlacementService instance."""
        if self._order_placement_service is None:
            bot = self._bot
            self._order_placement_service = OrderPlacementService(
                orders_store=bot.orders_store,
                order_stats=bot.order_stats,
                broker=bot.broker,
                dry_run=bot.config.dry_run,
                metrics_server=bot.metrics_server,
                guardrails=getattr(bot, "guardrails", None),
                profile=bot.config.profile.value,
            )
        return self._order_placement_service

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        """Execute decision using OrderPlacementService."""
        bot = self._bot
        service = self._get_order_placement_service()

        await service.execute_decision(
            symbol=symbol,
            decision=decision,
            mark=mark,
            product=product,
            position_state=position_state,
            exec_engine=bot.exec_engine,
            reduce_only_mode=bot.is_reduce_only_mode(),
            default_time_in_force=bot.config.time_in_force,
        )

    def ensure_order_lock(self) -> asyncio.Lock:
        """Ensure a shared asyncio.Lock exists for order placement."""
        bot = self._bot
        if getattr(bot, "_order_lock", None) is None:
            bot._order_lock = asyncio.Lock()

        service = self._get_order_placement_service()
        service._order_lock = bot._order_lock
        return bot._order_lock

    async def place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        """Place order while handling validation and execution failures."""
        lock = self.ensure_order_lock()

        stats = self._bot.order_stats
        stats.setdefault("failed", 0)

        try:
            async with lock:
                return await self._place_order_inner(exec_engine, **kwargs)
        except (ValidationError, ExecutionError, RiskValidationError) as exc:
            stats["failed"] += 1
            logger.warning("Order placement failed: %s", exc)
            raise
        except Exception as exc:
            stats["failed"] += 1
            logger.error("Unexpected error during order placement: %s", exc, exc_info=True)
            return None

    async def _place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        """Delegate to OrderPlacementService inner placement."""
        service = self._get_order_placement_service()
        return await service._place_order_inner(exec_engine, **kwargs)

    def _get_runtime_supervisor(self) -> ExecutionRuntimeSupervisor:
        """Get or create ExecutionRuntimeSupervisor instance."""
        if self._runtime_supervisor is None:
            bot = self._bot

            def _reconciler_factory() -> OrderReconciler:
                return OrderReconciler(
                    broker=bot.broker,
                    orders_store=bot.orders_store,
                    event_store=bot.event_store,
                    bot_id=bot.bot_id,
                )

            self._runtime_supervisor = ExecutionRuntimeSupervisor(
                exec_engine=bot.exec_engine,
                order_reconciler_factory=_reconciler_factory,
            )
        return self._runtime_supervisor

    def _get_order_reconciler(self) -> OrderReconciler:
        """Get order reconciler from supervisor."""
        if self._order_reconciler is None:
            supervisor = self._get_runtime_supervisor()
            self._order_reconciler = supervisor._get_order_reconciler()
        return self._order_reconciler

    def reset_order_reconciler(self) -> None:
        """Reset order reconciler (delegates to supervisor)."""
        if self._runtime_supervisor is not None:
            self._runtime_supervisor.reset_order_reconciler()
        self._order_reconciler = None

    async def run_runtime_guards(self) -> None:
        """Delegate to ExecutionRuntimeSupervisor."""
        supervisor = self._get_runtime_supervisor()
        await supervisor.run_runtime_guards(self._bot)

    async def run_order_reconciliation(self, interval_seconds: int = 45) -> None:
        """Delegate to ExecutionRuntimeSupervisor."""
        supervisor = self._get_runtime_supervisor()
        await supervisor.run_order_reconciliation(self._bot, interval_seconds)

    async def _run_order_reconciliation_cycle(
        self, reconciler: OrderReconciler | None = None
    ) -> None:
        """Execute a single reconciliation cycle for tests and manual triggering."""
        target_reconciler = reconciler or self._get_order_reconciler()
        local_open = target_reconciler.fetch_local_open_orders()
        exchange_open = await target_reconciler.fetch_exchange_open_orders()
        diff = target_reconciler.diff_orders(local_open, exchange_open)
        await target_reconciler.record_snapshot(local_open, exchange_open)

        if diff.missing_on_exchange:
            await target_reconciler.reconcile_missing_on_exchange(diff)
        if diff.missing_locally:
            target_reconciler.reconcile_missing_locally(diff)
