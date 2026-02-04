"""Order router for multi-venue execution.

Routes orders through TradingEngine.submit_order() via a submitter callable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from gpt_trader.core import Order, OrderSide
from gpt_trader.features.live_trade.execution.submission_result import (
    OrderSubmissionResult,
    OrderSubmissionStatus,
)
from gpt_trader.features.live_trade.strategies.hybrid.types import Action, HybridDecision
from gpt_trader.monitoring.metrics_collector import record_counter
from gpt_trader.utilities.datetime_helpers import utc_now
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="order_router")


def _resolve_order_id(order: Any | None) -> str | None:
    if order is None:
        return None
    return getattr(order, "id", None) or getattr(order, "order_id", None)


class SubmitterCallable(Protocol):
    """Callable protocol matching TradingEngine.submit_order."""

    def __call__(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        equity: Decimal,
        *,
        quantity_override: Decimal | None = None,
        reduce_only: bool = False,
        reason: str = "external_submission",
        confidence: float = 1.0,
    ) -> Awaitable[OrderSubmissionResult]: ...


@dataclass
class OrderResult:
    """Result of an order execution attempt."""

    success: bool
    order: Order | None = None
    error: str | None = None
    error_code: str | None = None
    decision: HybridDecision | None = None
    executed_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "order_id": _resolve_order_id(self.order),
            "symbol": (
                self.order.symbol
                if self.order
                else (self.decision.symbol if self.decision else None)
            ),
            "error": self.error,
            "error_code": self.error_code,
            "mode": self.decision.mode.value if self.decision else None,
            "executed_at": self.executed_at.isoformat(),
        }


class OrderRouter:
    """Routes orders through the canonical guard stack."""

    def __init__(
        self,
        *,
        submitter: SubmitterCallable,
        equity_provider: Callable[[], Decimal],
    ) -> None:
        """Initialize order router.

        Args:
            submitter: Async callable for canonical submission (TradingEngine.submit_order).
            equity_provider: Callable returning current equity (needed for submitter).
        """
        self._submitter = submitter
        self._equity_provider = equity_provider

    async def execute_async(self, decision: HybridDecision, price: Decimal) -> OrderResult:
        """Execute a hybrid decision through the canonical guard stack.

        All orders route through TradingEngine.submit_order() which applies
        the full pre-trade guard stack.
        """
        if not decision.is_actionable():
            return OrderResult(
                success=True,
                decision=decision,
                error="Decision is not actionable (HOLD)",
            )

        side = self._action_to_side(decision.action)
        if side is None:
            return OrderResult(
                success=False,
                decision=decision,
                error=f"Cannot convert action {decision.action} to order side",
                error_code="INVALID_ACTION",
            )

        equity = self._equity_provider()
        reduce_only = decision.action in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT)

        try:
            submission = await self._submitter(
                decision.symbol,
                side,
                price,
                equity,
                quantity_override=decision.quantity,
                reduce_only=reduce_only,
                reason=f"hybrid_{decision.mode.value}",
                confidence=float(decision.confidence) if decision.confidence else 1.0,
            )

            if submission.status is OrderSubmissionStatus.SUCCESS:
                logger.info(
                    "Executed via canonical path",
                    symbol=decision.symbol,
                    side=side.value,
                    quantity=str(decision.quantity),
                    mode=decision.mode.value,
                    operation="order_route",
                    stage="canonical_success",
                )
                return OrderResult(
                    success=True,
                    decision=decision,
                )

            if submission.status is OrderSubmissionStatus.BLOCKED:
                if submission.decision_trace is None:
                    try:
                        record_counter("gpt_trader_trades_blocked_total")
                    except Exception:
                        pass
                logger.warning(
                    "Order blocked via canonical path",
                    symbol=decision.symbol,
                    side=side.value,
                    quantity=str(decision.quantity),
                    mode=decision.mode.value,
                    reason=submission.reason,
                    operation="order_route",
                    stage="canonical_blocked",
                )
                return OrderResult(
                    success=False,
                    decision=decision,
                    error=submission.reason or "Order blocked",
                    error_code="ORDER_BLOCKED",
                )

            logger.error(
                "Order failed via canonical path",
                symbol=decision.symbol,
                error_message=submission.error,
                mode=decision.mode.value,
                operation="order_route",
                stage="canonical_failure",
            )
            return OrderResult(
                success=False,
                decision=decision,
                error=submission.error or "Order submission failed",
                error_code="EXECUTION_ERROR",
            )

        except Exception as e:
            logger.error(
                "Failed to execute via canonical path",
                symbol=decision.symbol,
                error_message=str(e),
                mode=decision.mode.value,
                operation="order_route",
                stage="canonical_failure",
            )
            return OrderResult(
                success=False,
                decision=decision,
                error=str(e),
                error_code="EXECUTION_ERROR",
            )

    def _action_to_side(self, action: Action) -> OrderSide | None:
        """Convert hybrid Action to OrderSide.

        Args:
            action: Hybrid action enum.

        Returns:
            OrderSide or None if action can't be converted.
        """
        mapping = {
            Action.BUY: OrderSide.BUY,
            Action.SELL: OrderSide.SELL,
            Action.CLOSE: OrderSide.SELL,
            Action.CLOSE_LONG: OrderSide.SELL,
            Action.CLOSE_SHORT: OrderSide.BUY,
        }
        return mapping.get(action)
