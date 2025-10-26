"""Exchange reconciliation helpers for runtime coordinator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bot_v2.orchestration.state_manager import ReduceOnlyModeSource

from .logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .coordinator import RuntimeCoordinator


class RuntimeCoordinatorReconcileMixin:
    """State reconciliation routines executed during startup."""

    async def reconcile_state_on_startup(self: RuntimeCoordinator) -> None:
        ctx = self.context
        config = ctx.config
        if config.dry_run or getattr(config, "perps_skip_startup_reconcile", False):
            logger.info(
                "Skipping startup reconciliation",
                reason="dry_run" if config.dry_run else "perps_skip_startup_reconcile",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        broker = ctx.broker or ctx.registry.broker
        if broker is None:
            logger.info(
                "No broker available for reconciliation; skipping",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        orders_store = ctx.orders_store
        event_store = ctx.event_store
        if orders_store is None or event_store is None:
            logger.warning(
                "Skipping reconciliation: missing orders/event store",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        logger.info(
            "Reconciling state with exchange",
            operation="startup_reconcile",
            stage="begin",
        )
        try:
            reconciler = self._order_reconciler_cls(
                broker=broker,
                orders_store=orders_store,
                event_store=event_store,
                bot_id=ctx.bot_id,
            )

            local_open = reconciler.fetch_local_open_orders()
            exchange_open = await reconciler.fetch_exchange_open_orders()

            logger.info(
                "Reconciliation snapshot",
                local_open=len(local_open),
                exchange_open=len(exchange_open),
                operation="startup_reconcile",
                stage="snapshot",
            )
            await reconciler.record_snapshot(local_open, exchange_open)

            diff = reconciler.diff_orders(local_open, exchange_open)
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

            try:
                snapshot = await reconciler.snapshot_positions()
                if snapshot:
                    runtime_state = ctx.runtime_state
                    if runtime_state is not None:
                        runtime_state.last_positions = snapshot
            except Exception as exc:
                logger.debug(
                    "Failed to snapshot initial positions",
                    error=str(exc),
                    exc_info=True,
                    operation="startup_reconcile",
                    stage="positions",
                )

            logger.info(
                "State reconciliation complete",
                operation="startup_reconcile",
                stage="complete",
            )
        except Exception as exc:
            logger.error(
                "Failed to reconcile state on startup",
                error=str(exc),
                exc_info=True,
                operation="startup_reconcile",
                stage="error",
            )
            try:
                if ctx.event_store is not None:
                    ctx.event_store.append_error(
                        bot_id=ctx.bot_id,
                        message="startup_reconcile_failed",
                        context={"error": str(exc)},
                    )
            except Exception:
                logger.exception(
                    "Failed to persist startup reconciliation error",
                    operation="startup_reconcile",
                    stage="error_persist",
                )
            state_manager = getattr(self.context.registry, "reduce_only_state_manager", None)
            if state_manager is not None:
                state_manager.set_reduce_only_mode(
                    enabled=True,
                    reason="startup_reconcile_failed",
                    source=ReduceOnlyModeSource.STARTUP_RECONCILE_FAILED,
                    metadata={"context": "startup_reconcile"},
                )
            else:
                self.set_reduce_only_mode(True, reason="startup_reconcile_failed")

    @property
    def _order_reconciler_cls(self):
        try:
            from bot_v2.orchestration.coordinators import runtime as runtime_pkg

            reconciler_cls = getattr(runtime_pkg, "OrderReconciler", None)
            if reconciler_cls is not None:
                return reconciler_cls
        except Exception:  # pragma: no cover - defensive guard for import issues
            pass

        from bot_v2.orchestration.order_reconciler import OrderReconciler

        return OrderReconciler


__all__ = ["RuntimeCoordinatorReconcileMixin"]
