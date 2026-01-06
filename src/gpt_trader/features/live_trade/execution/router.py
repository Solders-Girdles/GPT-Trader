"""Order router for multi-venue execution.

Routes orders to appropriate execution venues based on trading mode
(spot vs CFM futures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    TradingMode,
)
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

logger = get_logger(__name__, component="order_router")


@runtime_checkable
class OrderServiceProtocol(Protocol):
    """Protocol for order services."""

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> Order:
        """Place a new order."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...


@dataclass
class OrderResult:
    """Result of an order execution attempt."""

    success: bool
    order: Order | None = None
    error: str | None = None
    error_code: str | None = None
    decision: HybridDecision | None = None
    executed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "order_id": self.order.order_id if self.order else None,
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
    """Routes orders to appropriate execution venue.

    Handles execution routing for hybrid strategies that trade across
    spot and CFM futures markets.

    Features:
    - Routes based on TradingMode (spot, CFM, or both)
    - Applies risk checks before execution
    - Supports leverage for CFM orders
    - Handles reduce-only mode enforcement
    """

    def __init__(
        self,
        *,
        order_service: OrderServiceProtocol,
        risk_manager: LiveRiskManager | None = None,
    ) -> None:
        """Initialize order router.

        Args:
            order_service: Service for order execution.
            risk_manager: Optional risk manager for pre-trade checks.
        """
        self._order_service = order_service
        self._risk_manager = risk_manager

    def execute(self, decision: HybridDecision) -> OrderResult:
        """Execute a single hybrid decision.

        Args:
            decision: The trading decision to execute.

        Returns:
            OrderResult with execution outcome.
        """
        if not decision.is_actionable():
            return OrderResult(
                success=True,
                decision=decision,
                error="Decision is not actionable (HOLD)",
            )

        # Check risk manager reduce-only modes
        if self._risk_manager:
            if (
                decision.mode == TradingMode.CFM_ONLY
                and self._risk_manager.is_cfm_reduce_only_mode()
            ):
                if decision.action in (Action.BUY, Action.SELL) and not self._is_position_reducing(
                    decision
                ):
                    return OrderResult(
                        success=False,
                        decision=decision,
                        error="CFM reduce-only mode active",
                        error_code="CFM_REDUCE_ONLY",
                    )

            if self._risk_manager.is_reduce_only_mode():
                if not self._is_position_reducing(decision):
                    return OrderResult(
                        success=False,
                        decision=decision,
                        error="Global reduce-only mode active",
                        error_code="REDUCE_ONLY",
                    )

        # Route based on mode
        if decision.mode == TradingMode.SPOT_ONLY:
            return self._execute_spot(decision)
        elif decision.mode == TradingMode.CFM_ONLY:
            return self._execute_cfm(decision)
        else:
            # Hybrid mode - should not be used for single decision
            logger.warning("Received HYBRID mode decision for single execution")
            return self._execute_spot(decision)

    def execute_batch(self, decisions: list[HybridDecision]) -> list[OrderResult]:
        """Execute multiple decisions in order.

        Args:
            decisions: List of trading decisions to execute.

        Returns:
            List of OrderResults, one per decision.
        """
        results = []
        for decision in decisions:
            result = self.execute(decision)
            results.append(result)

            # Stop on first critical failure
            if not result.success and result.error_code in ("RISK_VIOLATION", "INSUFFICIENT_FUNDS"):
                logger.warning(
                    "Stopping batch execution due to critical failure: %s",
                    result.error,
                )
                break

        return results

    def _execute_spot(self, decision: HybridDecision) -> OrderResult:
        """Execute a spot market order.

        Args:
            decision: Spot trading decision.

        Returns:
            OrderResult with execution outcome.
        """
        try:
            side = self._action_to_side(decision.action)
            if side is None:
                return OrderResult(
                    success=False,
                    decision=decision,
                    error=f"Cannot convert action {decision.action} to order side",
                    error_code="INVALID_ACTION",
                )

            reduce_only = decision.action in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT)

            order = self._order_service.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=decision.quantity,
                reduce_only=reduce_only,
            )

            logger.info(
                "Executed spot order: symbol=%s, side=%s, qty=%s, order_id=%s",
                decision.symbol,
                side.value,
                decision.quantity,
                order.order_id,
            )

            return OrderResult(
                success=True,
                order=order,
                decision=decision,
            )

        except Exception as e:
            logger.error(
                "Failed to execute spot order: %s - %s",
                decision.symbol,
                str(e),
            )
            return OrderResult(
                success=False,
                decision=decision,
                error=str(e),
                error_code="EXECUTION_ERROR",
            )

    def _execute_cfm(self, decision: HybridDecision) -> OrderResult:
        """Execute a CFM futures order.

        Args:
            decision: CFM trading decision.

        Returns:
            OrderResult with execution outcome.
        """
        try:
            # Validate leverage if risk manager available
            if self._risk_manager and decision.leverage > 1:
                try:
                    self._risk_manager.validate_cfm_leverage(
                        decision.symbol,
                        decision.leverage,
                    )
                except Exception as e:
                    return OrderResult(
                        success=False,
                        decision=decision,
                        error=str(e),
                        error_code="LEVERAGE_EXCEEDED",
                    )

            side = self._action_to_side(decision.action)
            if side is None:
                return OrderResult(
                    success=False,
                    decision=decision,
                    error=f"Cannot convert action {decision.action} to order side",
                    error_code="INVALID_ACTION",
                )

            reduce_only = decision.action in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT)

            order = self._order_service.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=decision.quantity,
                reduce_only=reduce_only,
                leverage=decision.leverage if decision.leverage > 1 else None,
            )

            logger.info(
                "Executed CFM order: symbol=%s, side=%s, qty=%s, leverage=%dx, order_id=%s",
                decision.symbol,
                side.value,
                decision.quantity,
                decision.leverage,
                order.order_id,
            )

            return OrderResult(
                success=True,
                order=order,
                decision=decision,
            )

        except Exception as e:
            logger.error(
                "Failed to execute CFM order: %s - %s",
                decision.symbol,
                str(e),
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
            Action.CLOSE: OrderSide.SELL,  # Default to sell for close
            Action.CLOSE_LONG: OrderSide.SELL,
            Action.CLOSE_SHORT: OrderSide.BUY,
        }
        return mapping.get(action)

    def _is_position_reducing(self, decision: HybridDecision) -> bool:
        """Check if decision would reduce a position.

        Args:
            decision: The trading decision.

        Returns:
            True if this is a position-reducing action.
        """
        return decision.action in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT)
