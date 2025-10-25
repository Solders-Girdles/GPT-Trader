"""Order reconciliation functionality separated from execution coordinator.

This module contains the order reconciliation logic that was previously
embedded in the large execution.py file. It provides:

- Background order reconciliation loops
- Order status monitoring and updates
- Reconciliation conflict resolution
- Configurable reconciliation intervals
- Error handling and logging
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.coordinators.base import CoordinatorContext

from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="order_reconciliation")


class OrderReconciliationService:
    """Service responsible for order reconciliation and monitoring.

    This service runs background tasks to monitor order status
    and reconcile orders with the broker's current state.
    """

    def __init__(
        self,
        context: CoordinatorContext,
        order_reconciler: OrderReconciler,
        reconciliation_interval_seconds: int = 45,
    ) -> None:
        """Initialize the order reconciliation service.

        Args:
            context: Coordinator context with runtime configuration
            order_reconciler: Order reconciler instance to use
            reconciliation_interval_seconds: Interval between reconciliation cycles
        """
        self.context = context
        self.order_reconciler = order_reconciler
        self.reconciliation_interval_seconds = reconciliation_interval_seconds
        self._running = False
        self._reconciliation_task: asyncio.Task[Any] | None = None

    async def start_reconciliation(self) -> None:
        """Start the background order reconciliation process."""
        if self._running:
            logger.warning(
                "Order reconciliation already running",
                operation="order_reconciliation",
                status="already_running",
            )
            return

        self._running = True
        logger.info(
            "Starting order reconciliation",
            operation="order_reconciliation",
            interval_seconds=self.reconciliation_interval_seconds,
        )

        # Start the reconciliation loop
        self._reconciliation_task = asyncio.create_task(self._run_reconciliation_loop())

    async def stop_reconciliation(self) -> None:
        """Stop the background order reconciliation process."""
        if not self._running:
            logger.warning(
                "Order reconciliation not running",
                operation="order_reconciliation",
                status="not_running",
            )
            return

        self._running = False
        logger.info(
            "Stopping order reconciliation",
            operation="order_reconciliation",
        )

        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        self._reconciliation_task = None
        logger.info(
            "Order reconciliation stopped",
            operation="order_reconciliation",
            status="stopped",
        )

    def is_running(self) -> bool:
        """Check if the reconciliation service is running."""
        return self._running

    async def _run_reconciliation_loop(self) -> None:
        """Run the main reconciliation loop."""
        try:
            while self._running:
                logger.debug(
                    "Running reconciliation cycle",
                    operation="order_reconciliation_cycle",
                )

                # Run a single reconciliation cycle
                await self._run_reconciliation_cycle()

                # Wait for the next interval
                await asyncio.sleep(self.reconciliation_interval_seconds)

        except asyncio.CancelledError:
            logger.info(
                "Order reconciliation loop cancelled",
                operation="order_reconciliation",
            )
        except Exception as exc:
            logger.error(
                "Order reconciliation loop error",
                operation="order_reconciliation",
                error=str(exc),
                exc_info=True,
            )
            if self._running:  # Only restart if not intentionally stopped
                logger.info(
                    "Restarting reconciliation loop after error",
                    operation="order_reconciliation",
                )
                await asyncio.sleep(5)  # Brief delay before restart
                # Loop will continue with next iteration

    async def _run_reconciliation_cycle(self) -> None:
        """Run a single reconciliation cycle."""
        try:
            # Run reconciliation with the configured reconciler
            await self.order_reconciler.reconcile_orders()

            logger.debug(
                "Reconciliation cycle completed",
                operation="order_reconciliation_cycle",
                reconciler_type=type(self.order_reconciler).__name__,
            )

        except Exception as exc:
            logger.error(
                "Reconciliation cycle failed",
                operation="order_reconciliation_cycle",
                reconciler_type=type(self.order_reconciler).__name__,
                error=str(exc),
                exc_info=True,
            )
            # Continue with next cycle rather than stopping the entire service
            return

    async def force_reconciliation(self) -> None:
        """Force an immediate reconciliation cycle."""
        logger.info(
            "Forcing immediate reconciliation",
            operation="order_reconciliation_force",
        )

        try:
            await self._run_reconciliation_cycle()
            logger.info(
                "Forced reconciliation completed",
                operation="order_reconciliation_force",
            )
        except Exception as exc:
            logger.error(
                "Forced reconciliation failed",
                operation="order_reconciliation_force",
                error=str(exc),
                exc_info=True,
            )

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the reconciliation service."""
        return {
            "running": self.is_running(),
            "reconciliation_interval_seconds": self.reconciliation_interval_seconds,
            "reconciliation_task": self._reconciliation_task is not None,
            "reconciler_type": type(self.order_reconciler).__name__,
        }


__all__ = [
    "OrderReconciliationService",
]
