"""
Order event recording for live trading execution.

This module centralizes all event recording, telemetry emission, and
monitoring logger interactions for order lifecycle events.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from gpt_trader.app.protocols import EventStoreProtocol
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.monitoring.system import LogLevel
from gpt_trader.monitoring.system import get_logger as get_monitoring_logger
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="order_event_recorder")


class OrderEventRecorder:
    """Handles event recording and telemetry for order lifecycle."""

    def __init__(
        self,
        event_store: EventStoreProtocol,
        bot_id: str,
    ) -> None:
        """
        Initialize event recorder.

        Args:
            event_store: Event store for persisting events
            bot_id: Bot identifier for tagging events
        """
        self._event_store = event_store
        self._bot_id = bot_id

    def record_preview(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        preview: dict[str, Any] | None,
    ) -> None:
        """Record order preview for analysis."""
        if preview is None:
            return
        emit_metric(
            self._event_store,
            self._bot_id,
            {
                "event_type": "order_preview",
                "symbol": symbol,
                "side": side.value,
                "order_type": order_type.value,
                "quantity": str(quantity),
                "price": str(price) if price is not None else "market",
                "preview": preview,
            },
            logger=get_monitoring_logger(),
        )
        try:
            get_monitoring_logger().log_event(
                level=LogLevel.INFO,
                event_type="order_preview",
                message="Order preview generated",
                component="TradingEngine",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
            )
        except Exception as exc:
            logger.error(
                "Failed to log order preview event",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="record_preview",
                symbol=symbol,
                side=side.value,
            )

    def record_rejection(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal | None,
        reason: str,
        *,
        client_order_id: str | None = None,
    ) -> None:
        """Record order rejection for analysis.

        Args:
            symbol: Trading symbol.
            side: Order side (buy/sell).
            quantity: Order quantity.
            price: Order price (None for market orders).
            reason: Rejection reason.
            client_order_id: Client order ID for tracking (optional).
        """
        # Use client_order_id as fallback order_id for logging consistency
        effective_order_id = client_order_id or ""

        logger.warning(
            "Order rejected: %s %s %s @ %s reason=%s",
            symbol,
            side,
            quantity,
            price or "market",
            reason,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(price) if price is not None else None,
            reason=reason,
            client_order_id=effective_order_id,
            operation="order_rejected",
            stage="record",
        )
        emit_metric(
            self._event_store,
            self._bot_id,
            {
                "event_type": "order_rejected",
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "price": str(price) if price is not None else "market",
                "reason": reason,
                "client_order_id": effective_order_id,
            },
            logger=get_monitoring_logger(),
        )
        try:
            get_monitoring_logger().log_order_status_change(
                order_id=effective_order_id,
                client_order_id=effective_order_id,
                from_status=None,
                to_status="REJECTED",
                reason=reason,
            )
        except Exception as exc:
            logger.error(
                "Failed to log order rejection status change",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="record_rejection",
                symbol=symbol,
                reason=reason,
            )

    def record_submission_attempt(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
    ) -> None:
        """Log the order submission attempt to monitoring."""
        try:
            get_monitoring_logger().log_order_submission(
                client_order_id=submit_id,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=float(quantity),
                price=float(price) if price is not None else None,
            )
        except Exception as exc:
            logger.error(
                "Failed to log order submission attempt",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="log_submission_attempt",
                submit_id=submit_id,
                symbol=symbol,
                side=side.value,
            )

    def record_success(
        self,
        order: Any,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        display_price: Any,
        reduce_only: bool,
    ) -> None:
        """Log successful order placement."""
        logger.info(
            "Order placed: %s %s %s @ %s (reduce_only=%s)",
            side.value,
            quantity,
            symbol,
            display_price,
            reduce_only,
            order_id=str(order.id),
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            price=float(display_price) if isinstance(display_price, Decimal) else display_price,
            reduce_only=reduce_only,
            operation="order_submit",
            stage="placed",
        )

    def record_trade_event(
        self,
        order: Any,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        submit_id: str,
    ) -> None:
        """Record trade event to event store and monitoring."""
        try:
            get_monitoring_logger().log_order_status_change(
                order_id=str(order.id),
                client_order_id=getattr(order, "client_order_id", submit_id),
                from_status=None,
                to_status=getattr(order, "status", "SUBMITTED"),
            )
        except Exception as exc:
            logger.error(
                "Failed to log order status change",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="record_trade_event",
                order_id=str(order.id),
                symbol=symbol,
            )

        try:
            trade_quantity = getattr(order, "quantity", quantity)
            trade_payload = {
                "order_id": order.id,
                "client_order_id": getattr(order, "client_order_id", submit_id),
                "symbol": symbol,
                "side": side.value,
                "quantity": str(trade_quantity),
                "price": str(order.price or price or effective_price or "market"),
                "status": getattr(order, "status", "SUBMITTED"),
            }
            self._event_store.append_trade(self._bot_id, trade_payload)
        except Exception as exc:
            logger.error(
                "Failed to record trade event to event store",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="record_trade_event",
                order_id=str(order.id),
                symbol=symbol,
                side=side.value,
            )

    def record_failure(
        self,
        exc: Exception,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Record order placement failure."""
        logger.error(
            "Order placement failed",
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            operation="order_submit",
            stage="failed",
        )
        try:
            self._event_store.append_error(
                bot_id=self._bot_id,
                message="order_placement_failed",
                context={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                },
            )
        except Exception as store_exc:
            logger.error(
                "Failed to record order failure to event store",
                error_type=type(store_exc).__name__,
                error_message=str(store_exc),
                operation="handle_order_failure",
                symbol=symbol,
                side=side.value,
            )

    def record_broker_rejection(
        self,
        order: Any,
        status_name: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
    ) -> None:
        """Record broker rejection for non-integration mode."""
        self.record_rejection(
            symbol,
            side.value,
            quantity,
            price if price is not None else effective_price,
            f"broker_status_{status_name}",
        )
        try:
            self._event_store.append_error(
                bot_id=self._bot_id,
                message="broker_order_rejected",
                context={
                    "symbol": symbol,
                    "status": str(status_name),
                    "quantity": str(quantity),
                },
            )
        except Exception as exc:
            logger.error(
                "Failed to record order rejection to event store",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="process_rejection",
                symbol=symbol,
                status=str(status_name),
            )

    def record_integration_rejection(
        self,
        order: Any,
        symbol: str,
        status_name: str,
    ) -> None:
        """Record rejection in integration mode (simplified)."""
        self._event_store.store_event(
            "order_rejected",
            {
                "order_id": order.id,
                "symbol": symbol,
                "status": str(status_name),
            },
        )
