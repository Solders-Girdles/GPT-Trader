"""Background task management for the execution coordinator."""

from __future__ import annotations

import asyncio
from typing import Any

from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.utilities.async_utils import run_in_thread

from .logging_utils import logger


class ExecutionCoordinatorBackgroundMixin:
    """Encapsulate background task orchestration for execution coordination."""

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        ctx = self.context
        if ctx.config.dry_run or ctx.runtime_state is None:
            return []

        tasks: list[asyncio.Task[Any]] = []
        guards_task = asyncio.create_task(self._run_runtime_guards_loop())
        reconciliation_task = asyncio.create_task(self._run_order_reconciliation_loop())

        for task in (guards_task, reconciliation_task):
            self._register_background_task(task)
            tasks.append(task)

        return tasks

    async def run_runtime_guards(self) -> None:
        await self._run_runtime_guards_loop()

    async def _run_runtime_guards_loop(self) -> None:
        try:
            while True:
                exec_engine = getattr(self.context.runtime_state, "exec_engine", None)
                if exec_engine is not None:
                    try:
                        await run_in_thread(exec_engine.run_runtime_guards)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.error(
                            "Error in runtime guards",
                            error=str(exc),
                            exc_info=True,
                            operation="runtime_guard_loop",
                            stage="run",
                        )
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    async def run_order_reconciliation(self, interval_seconds: int = 45) -> None:
        await self._run_order_reconciliation_loop(interval_seconds=interval_seconds)

    async def _run_order_reconciliation_loop(self, interval_seconds: int = 45) -> None:
        try:
            while True:
                try:
                    reconciler = self._get_order_reconciler()
                    await self._run_order_reconciliation_cycle(reconciler)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Order reconciliation error",
                        error=str(exc),
                        exc_info=True,
                        operation="order_reconcile_loop",
                        stage="run",
                    )
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise

    async def _run_order_reconciliation_cycle(self, reconciler: OrderReconciler) -> None:
        orders_store = self.context.orders_store

        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()

        if len(local_open) != len(exchange_open):
            logger.info(
                "Order count mismatch",
                local=len(local_open),
                exchange=len(exchange_open),
                operation="order_reconcile",
                stage="diff",
            )

        diff = reconciler.diff_orders(local_open, exchange_open)
        if diff.missing_on_exchange or diff.missing_locally:
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

        if orders_store is not None:
            for order in exchange_open.values():
                try:
                    orders_store.upsert(order)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to upsert exchange order during reconciliation",
                        order_id=order.id,
                        error=str(exc),
                        exc_info=True,
                        operation="order_reconcile",
                        stage="upsert",
                    )

        await reconciler.record_snapshot(local_open, exchange_open)

    def _get_order_reconciler(self) -> OrderReconciler:
        if self._order_reconciler is None:
            ctx = self.context
            broker = ctx.broker
            if broker is None or ctx.orders_store is None or ctx.event_store is None:
                raise RuntimeError(
                    "Cannot create OrderReconciler without broker, stores, and event log"
                )
            self._order_reconciler = OrderReconciler(
                broker=broker,
                orders_store=ctx.orders_store,
                event_store=ctx.event_store,
                bot_id=ctx.bot_id,
            )
        return self._order_reconciler

    def reset_order_reconciler(self) -> None:
        self._order_reconciler = None


__all__ = ["ExecutionCoordinatorBackgroundMixin"]
