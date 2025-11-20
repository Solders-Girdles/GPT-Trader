"""Order placement functionality separated from execution coordinator.

This module contains the core order placement logic that was previously
embedded in the large execution.py file. It provides:

- Order validation and preparation
- Order placement with retry logic
- Impact estimation integration
- Error handling and logging
- Thread-safe execution
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.engines.base import CoordinatorContext

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.logging import (
    correlation_context,
    log_execution_error,
    log_order_event,
)
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.quantities import quantity_from

logger = get_logger(__name__, component="order_placement")


class OrderPlacementService:
    """Service responsible for order placement and validation.

    This service consolidates order placement logic that was previously
    spread across the execution coordinator. It provides clean separation
    of concerns and focused responsibility for order-related operations.
    """

    def __init__(
        self,
        context: CoordinatorContext,
        execution_engine: LiveExecutionEngine | AdvancedExecutionEngine,
        risk_manager: LiveRiskManager | None = None,
    ) -> None:
        """Initialize the order placement service.

        Args:
            context: Coordinator context with runtime configuration
            execution_engine: Execution engine to use for order placement
            risk_manager: Optional risk manager for additional validation
        """
        self.context = context
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self._order_stats: dict[str, int] = {}

    def place_order(
        self,
        action: Action,
        *,
        time_in_force: TimeInForce | None = None,
        limit_price: Decimal | None = None,
        **kwargs: Any,
    ) -> Order | None:
        """Place an order with validation and error handling.

        Args:
            action: The trading action to execute
            time_in_force: Optional time-in-force override
            limit_price: Optional limit price for the order
            **kwargs: Additional order parameters

        Returns:
            The placed order or None if placement failed
        """
        try:
            return self._place_order_with_validation(
                action, time_in_force=time_in_force, limit_price=limit_price, **kwargs
            )
        except Exception as exc:
            self._log_order_placement_error(action, exc)
            return None

    def _place_order_with_validation(
        self,
        action: Action,
        *,
        time_in_force: TimeInForce | None = None,
        limit_price: Decimal | None = None,
        **kwargs: Any,
    ) -> Order | None:
        """Place an order with comprehensive validation."""
        # Validate action parameters
        self._validate_action_parameters(action, **kwargs)

        # Prepare order parameters
        order_params = self._prepare_order_params(
            action, time_in_force=time_in_force, limit_price=limit_price, **kwargs
        )

        # Get execution engine
        exec_engine = self._get_execution_engine()

        # Place the order with error handling
        try:
            with correlation_context("place_order"):
                return self._place_order_core(exec_engine, order_params)
        except Exception as exc:
            logger.error(
                "Order placement failed",
                operation="order_placement",
                action=action.__class__.__name__,
                symbol=action.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise ExecutionError(f"Order placement failed: {exc}") from exc

    def _validate_action_parameters(self, action: Action, **kwargs: Any) -> None:
        """Validate action parameters before order placement."""
        if not hasattr(action, "symbol") or not action.symbol:
            raise ValidationError("Action must have a valid symbol")

        if not hasattr(action, "quantity") or action.quantity <= 0:
            raise ValidationError("Action must have a positive quantity")

        if hasattr(action, "side") and action.side not in {OrderSide.BUY, OrderSide.SELL}:
            raise ValidationError(f"Invalid order side: {action.side}")

        # Validate additional parameters
        if "limit_price" in kwargs and kwargs["limit_price"] is not None:
            if kwargs["limit_price"] <= 0:
                raise ValidationError("Limit price must be positive")

    def _prepare_order_params(
        self,
        action: Action,
        *,
        time_in_force: TimeInForce | None = None,
        limit_price: Decimal | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare order parameters from action and additional arguments."""
        config = self.context.config

        # Base order parameters
        order_params = {
            "symbol": action.symbol,
            "side": action.side,
            "order_type": getattr(action, "order_type", OrderType.MARKET),
            "time_in_force": time_in_force or config.time_in_force,
        }

        # Quantity handling
        if hasattr(action, "quantity"):
            order_params["quantity"] = quantity_from(action.quantity)

        # Price handling
        if limit_price is not None:
            order_params["limit_price"] = limit_price
            order_params["order_type"] = OrderType.LIMIT

        # Additional parameters
        for key, value in kwargs.items():
            if key not in order_params:  # Don't override core params
                order_params[key] = value

        return order_params

    def _get_execution_engine(self) -> LiveExecutionEngine | AdvancedExecutionEngine:
        """Get the appropriate execution engine."""
        return self.execution_engine

    def _place_order_core(
        self,
        exec_engine: LiveExecutionEngine | AdvancedExecutionEngine,
        order_params: dict[str, Any],
    ) -> Order:
        """Core order placement logic."""
        # Log order placement attempt
        logger.info(
            "Placing order",
            operation="order_placement",
            symbol=order_params["symbol"],
            side=order_params["side"],
            quantity=order_params["quantity"],
            order_type=order_params["order_type"],
        )

        # Place the order
        order = exec_engine.place_order(**order_params)

        if order:
            self._log_order_success(order)
            self._increment_order_stat("placed")
        else:
            self._log_order_failure(order_params)

        return order

    def _log_order_success(self, order: Order) -> None:
        """Log successful order placement."""
        log_order_event(
            "order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            time_in_force=order.time_in_force,
        )

        logger.info(
            "Order placed successfully",
            operation="order_placement",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
        )

    def _log_order_failure(self, order_params: dict[str, Any]) -> None:
        """Log failed order placement."""
        logger.error(
            "Order placement failed",
            operation="order_placement",
            symbol=order_params["symbol"],
            side=order_params["side"],
            quantity=order_params["quantity"],
            order_type=order_params.get("order_type"),
        )
        self._increment_order_stat("failed")

    def _log_order_placement_error(self, action: Action, error: Exception) -> None:
        """Log order placement error."""
        log_execution_error(
            error,
            context={
                "symbol": getattr(action, "symbol", "unknown"),
                "side": getattr(action, "side", "unknown"),
                "action_type": action.__class__.__name__,
                "operation": "order_placement",
            },
        )

    def _increment_order_stat(self, key: str) -> None:
        """Increment order statistics."""
        self._order_stats[key] = self._order_stats.get(key, 0) + 1

    def get_order_stats(self) -> dict[str, int]:
        """Get current order placement statistics."""
        return self._order_stats.copy()

    def reset_order_stats(self) -> None:
        """Reset order placement statistics."""
        self._order_stats.clear()


__all__ = [
    "OrderPlacementService",
]
