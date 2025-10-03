"""Supervisor for execution runtime background tasks.

Responsibilities:
- Run runtime guards loop
- Run order reconciliation loop
- Manage background task lifecycle
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.order_reconciler import OrderReconciler

logger = logging.getLogger(__name__)


class ExecutionRuntimeSupervisor:
    """Supervisor for execution runtime background tasks."""

    def __init__(
        self,
        exec_engine: Any,
        order_reconciler_factory: Any = None,
    ) -> None:
        """Initialize runtime supervisor.

        Args:
            exec_engine: Execution engine with run_runtime_guards method
            order_reconciler_factory: Callable that returns OrderReconciler instance
        """
        self._exec_engine = exec_engine
        self._order_reconciler_factory = order_reconciler_factory
        self._order_reconciler: OrderReconciler | None = None

    def _get_order_reconciler(self) -> OrderReconciler:
        """Get or create order reconciler instance."""
        if self._order_reconciler is None:
            if self._order_reconciler_factory is None:
                raise RuntimeError("OrderReconciler factory not configured")
            self._order_reconciler = self._order_reconciler_factory()
        return self._order_reconciler

    def reset_order_reconciler(self) -> None:
        """Reset order reconciler (useful when config changes)."""
        self._order_reconciler = None

    async def run_runtime_guards(self, running_flag: Any) -> None:
        """Run runtime guards loop.

        Args:
            running_flag: Object with .running attribute (bool)
        """
        await self._run_runtime_guards_loop(running_flag)

    async def _run_runtime_guards_loop(self, running_flag: Any) -> None:
        """Background loop for runtime guards."""
        while running_flag.running:
            try:
                await asyncio.to_thread(self._exec_engine.run_runtime_guards)
            except Exception as e:
                logger.error(f"Error in runtime guards: {e}", exc_info=True)
            await asyncio.sleep(60)

    async def run_order_reconciliation(self, running_flag: Any, interval_seconds: int = 45) -> None:
        """Run order reconciliation loop.

        Args:
            running_flag: Object with .running attribute (bool)
            interval_seconds: Sleep interval between reconciliation cycles
        """
        await self._run_order_reconciliation_loop(running_flag, interval_seconds)

    async def _run_order_reconciliation_loop(
        self, running_flag: Any, interval_seconds: int = 45
    ) -> None:
        """Background loop for order reconciliation."""
        while running_flag.running:
            try:
                reconciler = self._get_order_reconciler()
                await self._run_order_reconciliation_cycle(reconciler)
            except Exception as e:
                logger.debug(f"Order reconciliation error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)

    async def _run_order_reconciliation_cycle(self, reconciler: OrderReconciler) -> None:
        """Single order reconciliation cycle."""
        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()

        if len(local_open) != len(exchange_open):
            logger.info(
                "Order count mismatch: local=%s exchange=%s",
                len(local_open),
                len(exchange_open),
            )

        diff = reconciler.diff_orders(local_open, exchange_open)
        if diff.missing_on_exchange or diff.missing_locally:
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

        for order in exchange_open.values():
            try:
                # Note: This requires access to orders_store, which could be
                # passed via reconciler or injected here
                if hasattr(reconciler, "_orders_store"):
                    reconciler._orders_store.upsert(order)
            except Exception as exc:
                logger.debug(
                    "Failed to upsert exchange order %s during reconciliation: %s",
                    order.id,
                    exc,
                    exc_info=True,
                )

        await reconciler.record_snapshot(local_open, exchange_open)
