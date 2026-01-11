"""
Order submission and recording for live trading execution.

This module handles the actual submission of orders to the broker and
recording of order events, previews, and rejections.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, cast

from gpt_trader.app.protocols import EventStoreProtocol
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor
from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder
from gpt_trader.features.live_trade.execution.rejection_reason import (
    normalize_rejection_reason,
)
from gpt_trader.logging.correlation import order_context
from gpt_trader.monitoring.metrics_collector import record_counter, record_histogram
from gpt_trader.observability.tracing import trace_span
from gpt_trader.persistence.orders_store import (
    OrderRecord,
    OrdersStore,
)
from gpt_trader.persistence.orders_store import (
    OrderStatus as StoreOrderStatus,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="order_submission")


def _classify_rejection_reason(status_or_error: str) -> str:
    """Classify a rejection or error into a standardized reason category."""
    return normalize_rejection_reason(status_or_error)[0]


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
        rejection_reason: Categorized reason for rejection/failure (normalized).
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
        reason: Rejection/failure reason or "none" for success (normalized)
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


def _record_order_submission_latency(
    latency_seconds: float,
    result: str,
    side: str,
) -> None:
    """Record order submission latency histogram.

    Args:
        latency_seconds: Total submission latency in seconds.
        result: "success", "rejected", or "failed"
        side: "buy" or "sell"
    """
    try:
        record_histogram(
            "gpt_trader_order_submission_latency_seconds",
            latency_seconds,
            labels={
                "result": result,
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
        orders_store: OrdersStore | None = None,
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
        self.orders_store = orders_store
        self._event_recorder = OrderEventRecorder(event_store, bot_id)
        self._broker_executor = BrokerExecutor(
            cast(BrokerProtocol, broker), integration_mode=integration_mode
        )
        if self.orders_store is not None:
            try:
                self.orders_store.initialize()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize orders store",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="orders_store_init",
                )

    @staticmethod
    def _normalize_side(side: OrderSide | str) -> str:
        value = side.value if hasattr(side, "value") else str(side)
        return value.lower()

    @staticmethod
    def _normalize_order_type(order_type: OrderType | str) -> str:
        value = order_type.value if hasattr(order_type, "value") else str(order_type)
        return value.lower()

    @staticmethod
    def _normalize_status(status: Any) -> StoreOrderStatus:
        value = status.value if hasattr(status, "value") else status
        normalized = str(value).lower()
        mapping = {
            "pending": StoreOrderStatus.PENDING,
            "submitted": StoreOrderStatus.OPEN,
            "open": StoreOrderStatus.OPEN,
            "partially_filled": StoreOrderStatus.PARTIALLY_FILLED,
            "filled": StoreOrderStatus.FILLED,
            "cancelled": StoreOrderStatus.CANCELLED,
            "canceled": StoreOrderStatus.CANCELLED,
            "rejected": StoreOrderStatus.REJECTED,
            "expired": StoreOrderStatus.EXPIRED,
            "failed": StoreOrderStatus.FAILED,
        }
        return mapping.get(normalized, StoreOrderStatus.OPEN)

    @staticmethod
    def _normalize_tif(tif: Any) -> str:
        if tif is None:
            return "GTC"
        value = tif.value if hasattr(tif, "value") else tif
        return str(value)

    @staticmethod
    def _normalize_timestamp(value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _ensure_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _persist_order(self, record: OrderRecord) -> None:
        if self.orders_store is None:
            return
        try:
            self.orders_store.upsert_by_client_id(record)
        except Exception as exc:
            logger.warning(
                "Failed to persist order record",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="orders_store_persist",
                order_id=record.order_id,
                client_order_id=record.client_order_id,
            )

    def _build_submission_record(
        self,
        *,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal,
        price: Decimal | None,
        status: StoreOrderStatus,
        filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None,
        tif: Any | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> OrderRecord:
        created = self._normalize_timestamp(created_at)
        updated = self._normalize_timestamp(updated_at or created)
        return OrderRecord(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=self._normalize_side(side),
            order_type=self._normalize_order_type(order_type),
            quantity=quantity,
            price=price,
            status=status,
            filled_quantity=filled_quantity if filled_quantity is not None else Decimal("0"),
            average_fill_price=average_fill_price,
            created_at=created,
            updated_at=updated,
            bot_id=self.bot_id,
            time_in_force=self._normalize_tif(tif),
            metadata=metadata,
        )

    def _record_pending_submission(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> None:
        metadata: dict[str, Any] = {"reduce_only": reduce_only}
        if leverage is not None:
            metadata["leverage"] = leverage
        record = self._build_submission_record(
            order_id=submit_id,
            client_order_id=submit_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            status=StoreOrderStatus.PENDING,
            tif=tif,
            metadata=metadata,
        )
        self._persist_order(record)

    def _record_final_submission(
        self,
        order: Any,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        tif: Any | None,
        status: StoreOrderStatus,
        reduce_only: bool,
        leverage: int | None,
    ) -> None:
        metadata: dict[str, Any] = {"reduce_only": reduce_only}
        if leverage is not None:
            metadata["leverage"] = leverage
        record = self._build_submission_record(
            order_id=str(getattr(order, "id", submit_id)),
            client_order_id=str(getattr(order, "client_id", submit_id) or submit_id),
            symbol=str(getattr(order, "symbol", symbol)),
            side=getattr(order, "side", side),
            order_type=getattr(order, "type", order_type),
            quantity=self._ensure_decimal(getattr(order, "quantity", order_quantity)),
            price=getattr(order, "price", price),
            status=status,
            filled_quantity=self._ensure_decimal(getattr(order, "filled_quantity", None)),
            average_fill_price=getattr(order, "avg_fill_price", None),
            tif=getattr(order, "tif", tif),
            metadata=metadata,
            created_at=getattr(order, "created_at", None) or getattr(order, "submitted_at", None),
            updated_at=getattr(order, "updated_at", None),
        )
        self._persist_order(record)

    def _record_failed_submission(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
        status: StoreOrderStatus = StoreOrderStatus.FAILED,
    ) -> None:
        metadata: dict[str, Any] = {"reduce_only": reduce_only}
        if leverage is not None:
            metadata["leverage"] = leverage
        record = self._build_submission_record(
            order_id=submit_id,
            client_order_id=submit_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            status=status,
            tif=tif,
            metadata=metadata,
        )
        self._persist_order(record)

    def generate_client_order_id(self, client_order_id: str | None = None) -> str:
        """Generate a stable client order ID using submission rules."""
        return self._generate_submit_id(client_order_id)

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
        self._event_recorder.record_rejection(
            symbol, side, quantity, price, reason, client_order_id=client_order_id
        )

    def record_decision_trace(self, trace: OrderDecisionTrace) -> None:
        """Record decision trace for later analysis."""
        self._event_recorder.record_decision_trace(trace)

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
                    "bot_id": self.bot_id,
                    "client_order_id": submit_id,
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
        self._record_pending_submission(
            submit_id,
            symbol,
            side,
            order_type,
            order_quantity,
            price,
            tif,
            reduce_only,
            leverage,
        )

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
                order,
                symbol,
                side,
                order_type,
                order_quantity,
                price,
                effective_price,
                tif,
                reduce_only,
                leverage,
                submit_id,
            )

            # Record telemetry for successful submission
            side_str = side.value if hasattr(side, "value") else str(side)
            price_float = float(price) if price is not None else 0.0
            qty_float = float(order_quantity)

            # Convert latency to seconds for histogram
            latency_seconds = latency_ms / 1000.0

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
                _record_order_submission_latency(
                    latency_seconds=latency_seconds,
                    result="success",
                    side=side_str,
                )
            else:
                # Order was rejected by broker (fallback case - no order returned)
                _record_execution_telemetry(
                    latency_ms=latency_ms,
                    success=False,
                    rejected=True,
                    rejection_reason="broker_rejected",
                    symbol=symbol,
                    side=side_str,
                    quantity=qty_float,
                    price=price_float,
                )
                _record_order_submission_metric(
                    result="rejected",
                    reason="broker_rejected",
                    side=side_str,
                )
                _record_order_submission_latency(
                    latency_seconds=latency_seconds,
                    result="rejected",
                    side=side_str,
                )

            return result

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Classify the error for telemetry
            error_str = str(exc)
            reason = _classify_rejection_reason(error_str)
            is_rejection = reason in {"broker_rejected", "broker_status"}

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
            _record_order_submission_latency(
                latency_seconds=latency_ms / 1000.0,
                result=metric_result,
                side=side_str,
            )
            # 4. Handle Failure
            self._record_failed_submission(
                submit_id,
                symbol,
                side,
                order_type,
                order_quantity,
                price,
                tif,
                reduce_only,
                leverage,
            )
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
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
        submit_id: str,
    ) -> str | None:
        """Process the result from the broker, handling rejections and successes."""
        if not (order and order.id):
            # Should have raised exception if failed, but handle None case just in case
            self._record_failed_submission(
                submit_id,
                symbol,
                side,
                order_type,
                quantity,
                price,
                tif,
                reduce_only,
                leverage,
                status=StoreOrderStatus.REJECTED,
            )
            return None

        status_value = getattr(order, "status", None)
        if status_value is not None and hasattr(status_value, "value"):
            status_name = status_value.value
        else:
            status_name = str(status_value or "")
        store_status = self._normalize_status(status_value)

        # Check for Rejection
        if str(status_name).upper() in {"REJECTED", "CANCELLED", "FAILED"}:
            return cast(
                str | None,
                self._process_rejection(
                    order,
                    status_name,
                    symbol,
                    side,
                    order_type,
                    quantity,
                    price,
                    effective_price,
                    tif,
                    reduce_only,
                    leverage,
                    submit_id,
                    store_status,
                ),
            )

        # Process Success
        self.open_orders.append(order.id)
        display_price = price if price is not None else "market"

        self._log_success(order, symbol, side, quantity, display_price, reduce_only)
        self._record_trade_event(order, symbol, side, quantity, price, effective_price, submit_id)
        self._record_final_submission(
            order,
            submit_id,
            symbol,
            side,
            order_type,
            quantity,
            price,
            tif,
            store_status,
            reduce_only,
            leverage,
        )

        return cast(str | None, order if self.integration_mode else order.id)

    def _process_rejection(
        self,
        order: Any,
        status_name: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
        submit_id: str,
        store_status: StoreOrderStatus,
    ) -> Any:
        """Handle rejected orders."""
        self._record_final_submission(
            order,
            submit_id,
            symbol,
            side,
            order_type,
            quantity,
            price,
            tif,
            store_status,
            reduce_only,
            leverage,
        )
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
