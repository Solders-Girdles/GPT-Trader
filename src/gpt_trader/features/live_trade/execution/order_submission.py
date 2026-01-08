"""
Order submission and recording for live trading execution.

This module handles the actual submission of orders to the broker and
recording of order events, previews, and rejections.
"""

from __future__ import annotations

import os
import time
import uuid
from decimal import Decimal
from typing import Any, cast

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder
from gpt_trader.logging.correlation import order_context
from gpt_trader.monitoring.metrics_collector import record_counter
from gpt_trader.observability.tracing import trace_span
from gpt_trader.orchestration.protocols import EventStoreProtocol
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="order_submission")


def _classify_rejection_reason(status_or_error: str) -> str:
    """Classify a rejection or error into a standardized reason category.

    Args:
        status_or_error: The status string or error message.

    Returns:
        Standardized reason: rate_limit, insufficient_funds, invalid_size,
        invalid_price, market_closed, timeout, network, rejected, failed, unknown.
    """
    text = status_or_error.lower()

    # Rate limiting
    if any(term in text for term in ["rate_limit", "rate limit", "429", "too many"]):
        return "rate_limit"

    # Insufficient funds
    if any(term in text for term in ["insufficient", "balance", "funds", "margin"]):
        return "insufficient_funds"

    # Invalid size
    if any(term in text for term in ["size", "quantity", "min_", "max_", "amount"]):
        return "invalid_size"

    # Invalid price
    if any(term in text for term in ["price", "tick", "increment"]):
        return "invalid_price"

    # Timeout
    if any(term in text for term in ["timeout", "timed out", "deadline"]):
        return "timeout"

    # Network issues (check before market_closed to avoid "socket closed" matching "closed")
    if any(term in text for term in ["connection", "network", "socket", "dns", "ssl"]):
        return "network"

    # Market closed
    if any(term in text for term in ["market closed", "trading halt", "suspended"]):
        return "market_closed"

    # Generic rejection/failure
    if "reject" in text:
        return "rejected"
    if "fail" in text or "error" in text:
        return "failed"

    return "unknown"


def _record_execution_telemetry(
    latency_ms: float,
    success: bool,
    rejected: bool = False,
    failure_reason: str = "",
    rejection_reason: str = "",
    symbol: str = "",
    side: str = "",
    quantity: float = 0.0,
    price: float = 0.0,
) -> None:
    """Record execution telemetry if TUI service is available.

    This is a best-effort operation - failures are silently ignored
    to avoid impacting order execution.

    Args:
        latency_ms: Time from submission to response.
        success: Whether the submission succeeded.
        rejected: Whether broker rejected the order.
        failure_reason: Human-readable failure reason.
        rejection_reason: Categorized reason for rejection/failure.
        symbol: Order symbol.
        side: Order side (BUY/SELL).
        quantity: Order quantity.
        price: Order price.
    """
    try:
        from gpt_trader.tui.services.execution_telemetry import get_execution_telemetry

        collector = get_execution_telemetry()
        collector.record_submission(
            latency_ms=latency_ms,
            success=success,
            rejected=rejected,
            failure_reason=failure_reason,
            rejection_reason=rejection_reason,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )
    except Exception:
        # Don't let telemetry errors affect order execution
        pass


def _record_order_submission_metric(
    result: str,
    reason: str,
    side: str,
) -> None:
    """Record order submission metric with labels.

    Args:
        result: "success", "rejected", "failed", or "error"
        reason: Rejection/failure reason or "none" for success
        side: "buy" or "sell"
    """
    try:
        record_counter(
            "gpt_trader_order_submission_total",
            labels={
                "result": result,
                "reason": reason,
                "side": side.lower(),
            },
        )
    except Exception:
        # Don't let metrics errors affect order execution
        pass


class OrderSubmitter:
    """Handles order submission and event recording."""

    def __init__(
        self,
        broker: BrokerProtocol,
        event_store: EventStoreProtocol,
        bot_id: str,
        open_orders: list[str],
        *,
        integration_mode: bool = False,
    ) -> None:
        """
        Initialize order submitter.

        Args:
            broker: Brokerage adapter
            event_store: Event store for recording
            bot_id: Bot identifier
            open_orders: List to track open order IDs
            integration_mode: Enable integration test mode
        """
        self.broker = broker
        self.event_store = event_store
        self.bot_id = bot_id
        self.open_orders = open_orders
        self.integration_mode = integration_mode
        self._event_recorder = OrderEventRecorder(event_store, bot_id)
        self._broker_executor = BrokerExecutor(
            cast(BrokerProtocol, broker), integration_mode=integration_mode
        )

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
        self._event_recorder.record_preview(symbol, side, order_type, quantity, price, preview)

    def record_rejection(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal | None, reason: str
    ) -> None:
        """Record order rejection for analysis."""
        self._event_recorder.record_rejection(symbol, side, quantity, price, reason)

    def submit_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
        client_order_id: str | None,
    ) -> str | None:
        """
        Submit order to broker and record the result.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            order_quantity: Order quantity
            price: Order price (None for market)
            effective_price: Effective price for recording
            stop_price: Stop price for stop orders
            tif: Time in force
            reduce_only: Whether order is reduce-only
            leverage: Leverage multiplier
            client_order_id: Client order ID (generated if None)

        Returns:
            Order ID if successful, None otherwise
        """
        submit_id = self._generate_submit_id(client_order_id)

        # Wrap entire submission in order context and trace span
        with order_context(order_id=submit_id, symbol=symbol):
            with trace_span(
                "order_submit",
                {
                    "symbol": symbol,
                    "side": side.value if hasattr(side, "value") else str(side),
                    "order_type": (
                        order_type.value if hasattr(order_type, "value") else str(order_type)
                    ),
                    "quantity": float(order_quantity),
                    "reduce_only": reduce_only,
                },
            ):
                return self._submit_order_inner(
                    submit_id=submit_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    order_quantity=order_quantity,
                    price=price,
                    effective_price=effective_price,
                    stop_price=stop_price,
                    tif=tif,
                    reduce_only=reduce_only,
                    leverage=leverage,
                )

    def _submit_order_inner(
        self,
        *,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> str | None:
        """Inner order submission logic wrapped in correlation context."""
        # 1. Log submission attempt
        self._log_submission_attempt(submit_id, symbol, side, order_type, order_quantity, price)

        start_time = time.perf_counter()
        try:
            # 2. Execute Broker Request
            order = self._execute_broker_order(
                submit_id=submit_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
                leverage=leverage,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            # 3. Handle Result (Success/Rejection)
            result = self._handle_order_result(
                order, symbol, side, order_quantity, price, effective_price, reduce_only, submit_id
            )

            # Record telemetry for successful submission
            side_str = side.value if hasattr(side, "value") else str(side)
            price_float = float(price) if price is not None else 0.0
            qty_float = float(order_quantity)

            if result is not None:
                _record_execution_telemetry(
                    latency_ms=latency_ms,
                    success=True,
                    symbol=symbol,
                    side=side_str,
                    quantity=qty_float,
                    price=price_float,
                )
                _record_order_submission_metric(
                    result="success",
                    reason="none",
                    side=side_str,
                )
            else:
                # Order was rejected by broker (fallback case - no order returned)
                _record_execution_telemetry(
                    latency_ms=latency_ms,
                    success=False,
                    rejected=True,
                    rejection_reason="rejected",
                    symbol=symbol,
                    side=side_str,
                    quantity=qty_float,
                    price=price_float,
                )
                _record_order_submission_metric(
                    result="rejected",
                    reason="rejected",
                    side=side_str,
                )

            return result

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Classify the error for telemetry
            error_str = str(exc)
            reason = _classify_rejection_reason(error_str)
            is_rejection = "rejected by broker" in error_str.lower()

            # Compute order context for telemetry
            side_str = side.value if hasattr(side, "value") else str(side)
            price_float = float(price) if price is not None else 0.0
            qty_float = float(order_quantity)

            # Record telemetry for failed submission
            _record_execution_telemetry(
                latency_ms=latency_ms,
                success=False,
                rejected=is_rejection,
                failure_reason=error_str[:100],
                rejection_reason=reason,
                symbol=symbol,
                side=side_str,
                quantity=qty_float,
                price=price_float,
            )
            # Record metric with rejection/failure classification
            metric_result = "rejected" if is_rejection else "failed"
            _record_order_submission_metric(
                result=metric_result,
                reason=reason,
                side=side_str,
            )
            # 4. Handle Failure
            self._handle_order_failure(exc, symbol, side, order_quantity)
            return None

    def _generate_submit_id(self, client_order_id: str | None) -> str:
        """Generate or retrieve client order ID."""
        submit_id = (
            client_order_id or f"{self.bot_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        )
        if self.integration_mode:
            forced_id = os.getenv("INTEGRATION_TEST_ORDER_ID")
            if forced_id:
                submit_id = forced_id
        return submit_id

    def _log_submission_attempt(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
    ) -> None:
        """Log the order submission attempt to monitoring."""
        self._event_recorder.record_submission_attempt(
            submit_id, symbol, side, order_type, quantity, price
        )

    def _execute_broker_order(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> Any:
        """Execute the order placement against the broker."""
        return self._broker_executor.execute_order(
            submit_id=submit_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

    def _handle_order_result(
        self,
        order: Any,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        reduce_only: bool,
        submit_id: str,
    ) -> str | None:
        """Process the result from the broker, handling rejections and successes."""
        if not (order and order.id):
            # Should have raised exception if failed, but handle None case just in case
            return None

        status_value = getattr(order, "status", None)
        if status_value is not None and hasattr(status_value, "value"):
            status_name = status_value.value
        else:
            status_name = str(status_value or "")

        # Check for Rejection
        if str(status_name).upper() in {"REJECTED", "CANCELLED", "FAILED"}:
            return cast(
                str | None,
                self._process_rejection(
                    order, status_name, symbol, side, quantity, price, effective_price
                ),
            )

        # Process Success
        self.open_orders.append(order.id)
        display_price = price if price is not None else "market"

        self._log_success(order, symbol, side, quantity, display_price, reduce_only)
        self._record_trade_event(order, symbol, side, quantity, price, effective_price, submit_id)

        return cast(str | None, order if self.integration_mode else order.id)

    def _process_rejection(
        self,
        order: Any,
        status_name: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
    ) -> Any:
        """Handle rejected orders."""
        if self.integration_mode:
            self._event_recorder.record_integration_rejection(order, symbol, status_name)
            return order

        self._event_recorder.record_broker_rejection(
            order, status_name, symbol, side, quantity, price, effective_price
        )
        raise RuntimeError(f"Order rejected by broker: {status_name}")

    def _log_success(
        self,
        order: Any,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        display_price: Any,
        reduce_only: bool,
    ) -> None:
        """Log successful order placement."""
        self._event_recorder.record_success(
            order, symbol, side, quantity, display_price, reduce_only
        )

    def _record_trade_event(
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
        self._event_recorder.record_trade_event(
            order, symbol, side, quantity, price, effective_price, submit_id
        )

    def _handle_order_failure(
        self,
        exc: Exception,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Handle exceptions during order placement."""
        self._event_recorder.record_failure(exc, symbol, side, quantity)
