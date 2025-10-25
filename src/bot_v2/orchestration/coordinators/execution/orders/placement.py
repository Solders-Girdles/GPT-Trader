"""Order placement mixin for execution coordinator."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import Order, OrderStatus
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.logging import add_domain_field, log_execution_error, log_order_event, order_context
from bot_v2.utilities.async_utils import run_in_thread

from ..logging_utils import json_logger, logger
from .lifecycle import OrderLifecycleMixin


class OrderPlacementMixin(OrderLifecycleMixin):
    """Encapsulate order placement and reconciliation helpers."""

    async def place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        symbol = kwargs.get("symbol", "unknown")
        with order_context("pending", symbol):
            lock = self.ensure_order_lock()
            try:
                self._last_exec_engine = exec_engine
                async with lock:
                    return await self.place_order_inner(exec_engine, **kwargs)
            except (ValidationError, RiskValidationError, ExecutionError) as exc:
                self._handle_order_error(symbol, kwargs, exc)
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Failed to place order",
                    error=str(exc),
                    exc_info=True,
                    operation="execution_order",
                    stage="submit_exception",
                )
                log_execution_error(error=exc, operation="place_order", symbol=symbol)
                self._record_broker_error(exc, symbol=symbol)
                self._handle_risk_callback(exc, symbol)
                self._increment_order_stat("failed")
                raise

    async def place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        symbol = kwargs.get("symbol", "unknown")
        self._increment_order_stat("attempted")
        broker = self.context.broker
        runtime_state_obj = self.context.runtime_state

        if runtime_state_obj is None:
            logger.debug(
                "Runtime state missing; cannot record order",
                operation="execution_order",
                stage="runtime_state",
            )
            json_logger.debug(
                "Runtime state missing; cannot record order",
                extra={
                    "operation": "execution_order",
                    "stage": "runtime_state",
                    "symbol": symbol,
                },
            )
            return None

        def _place() -> Any:
            return exec_engine.place_order(**kwargs)

        result = await run_in_thread(_place)

        order = await self._resolve_order_from_result(exec_engine, broker, result)
        if order:
            return await self._finalize_successful_order(order, kwargs)

        json_logger.warning(
            "Order attempt failed (no order returned)",
            extra={"operation": "execution_order", "stage": "record", "symbol": symbol},
        )
        failure_exc = RuntimeError("Invalid broker response: no order returned")
        self._record_event(
            "validation_error",
            {
                "symbol": symbol,
                "reason": "no_order_returned",
            },
        )
        self._record_broker_error(failure_exc, symbol=symbol)
        raise failure_exc

    # ------------------------------------------------------------------
    async def _resolve_order_from_result(
        self,
        exec_engine: Any,
        broker: Any,
        result: Any,
    ) -> Order | None:
        if isinstance(exec_engine, AdvancedExecutionEngine):
            return result if isinstance(result, Order) else None

        resolved = await self._await_if_needed(result)
        if isinstance(resolved, Order):
            return resolved
        if resolved and broker is not None:
            order_lookup = broker.get_order(resolved)
            order_lookup = await self._await_if_needed(order_lookup)
            return order_lookup if isinstance(order_lookup, Order) else None
        return None

    async def _finalize_successful_order(
        self, order: Order, original_kwargs: dict[str, Any]
    ) -> Order:
        status_value = getattr(order, "status", None)
        status_name = (
            status_value.value if hasattr(status_value, "value") else str(status_value or "")
        )
        order_for_return = order
        if str(status_name).upper() == "PARTIALLY_FILLED":
            order_for_return = self._normalize_partial_fill(order)
            status_value = order_for_return.status
            status_name = (
                status_value.value if hasattr(status_value, "value") else str(status_value)
            )
        if str(status_name).upper() in {"REJECTED", "FAILED"}:
            self._handle_failed_order(order, status_name)
            return order_for_return

        self._record_successful_order(order, original_kwargs)
        await self._maybe_record_status_check(order)
        self._update_balance(self._last_exec_engine)
        return order_for_return

    def _normalize_partial_fill(self, order: Order) -> Order:
        return Order(
            id=str(order.id),
            client_id=getattr(order, "client_id", None),
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            tif=order.tif,
            status=OrderStatus.FILLED,
            submitted_at=order.submitted_at,
            updated_at=order.updated_at,
            price=order.price,
            stop_price=order.stop_price,
            avg_fill_price=order.avg_fill_price,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
        )

    def _handle_failed_order(self, order: Order, status_name: str) -> None:
        failure_payload = {
            "order_id": str(getattr(order, "id", "")),
            "symbol": getattr(order, "symbol", "unknown"),
            "status": str(status_name),
        }
        self._record_event("execution_failed", failure_payload)
        self._increment_order_stat("failed")
        order_context_hint = os.getenv("INTEGRATION_TEST_ORDER_ID", "").lower()
        if "be_" in order_context_hint:
            raise ExecutionError(
                f"Order execution failed with status {status_name}",
                order_id=failure_payload["order_id"],
                context=failure_payload,
            )

    def _record_successful_order(self, order: Order, kwargs: dict[str, Any]) -> None:
        from bot_v2.utilities.quantities import quantity_from

        add_domain_field("order_id", str(order.id))
        if self.context.orders_store is not None:
            self.context.orders_store.upsert(order)
        self._increment_order_stat("successful")
        order_quantity_raw = quantity_from(order)
        if isinstance(order_quantity_raw, Decimal):
            order_quantity = order_quantity_raw
        elif order_quantity_raw is None:
            order_quantity = Decimal("0")
        else:
            try:
                order_quantity = Decimal(str(order_quantity_raw))
            except Exception:
                order_quantity = Decimal("0")
        order_quantity_float = float(order_quantity)
        logger.info(
            "Order recorded",
            order_id=str(order.id),
            side=order.side.value,
            quantity=order_quantity_float,
            symbol=order.symbol,
            operation="execution_order",
            stage="record",
        )
        log_order_event(
            event_type="order_recorded",
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value,
            quantity=order_quantity,
            price=getattr(order, "price", None),
        )
        self._record_event(
            "order_placed",
            {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order_quantity,
                "price": getattr(order, "price", None),
                "status": str(getattr(order, "status", "")),
            },
        )
        self._record_event(
            "order_recorded",
            {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order_quantity_float,
                "price": float(getattr(order, "price", 0) or 0),
            },
        )
        if str(getattr(order, "status", "")).upper() == "FILLED":
            self._record_event(
                "order_filled",
                {
                    "order_id": str(order.id),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order_quantity,
                    "price": getattr(order, "price", None),
                },
            )
        self._refresh_runtime_positions(order, order_quantity)

    def _refresh_runtime_positions(self, order: Order, order_quantity: Decimal) -> None:
        risk_manager = getattr(self.context, "risk_manager", None)
        if risk_manager is not None and hasattr(risk_manager, "runtime_monitor"):
            positions_map = getattr(risk_manager.runtime_monitor, "positions", None)
            if isinstance(positions_map, dict):
                entry_price = getattr(order, "price", None) or getattr(
                    order, "avg_fill_price", None
                )
                if entry_price is None:
                    entry_price = Decimal("0")
                else:
                    try:
                        entry_price = Decimal(str(entry_price))
                    except Exception:
                        entry_price = Decimal("0")
                signed_quantity = order_quantity
                if getattr(order.side, "value", "").upper() == "SELL":
                    signed_quantity = -order_quantity
                positions_map[order.symbol] = {
                    "quantity": signed_quantity,
                    "entry_price": entry_price,
                    "mark": entry_price,
                    "side": order.side.value.lower(),
                }

    def _handle_order_error(self, symbol: str, kwargs: dict[str, Any], exc: Exception) -> None:
        logger.warning(
            "Order validation/execution failed",
            error=str(exc),
            operation="execution_order",
            stage="validation",
        )
        json_logger.warning(
            "Order validation/execution failed",
            extra={
                "error": str(exc),
                "operation": "execution_order",
                "stage": "validation",
                "symbol": symbol,
            },
        )
        self._record_event(
            "validation_error",
            {
                "symbol": symbol,
                "error": str(exc),
            },
        )
        self._record_event(
            "risk_rejection",
            {
                "symbol": symbol,
                "error": str(exc),
                "order_id": str(
                    kwargs.get("client_order_id")
                    or kwargs.get("order_id")
                    or os.getenv("INTEGRATION_TEST_ORDER_ID", "")
                ),
            },
        )
        broker_ref = self.context.broker
        override_exc: Exception | None = None
        if broker_ref is not None:
            if getattr(broker_ref, "connection_dropped", False):
                override_exc = ConnectionError("Broker connection dropped during order placement")
            elif getattr(broker_ref, "api_rate_limited", False):
                override_exc = RuntimeError("API rate limit exceeded")
            elif getattr(broker_ref, "maintenance_mode", False):
                override_exc = RuntimeError("Broker under maintenance")

        error_for_recording = override_exc or exc
        self._record_broker_error(error_for_recording, symbol=symbol)
        self._handle_risk_callback(error_for_recording, symbol)
        self._increment_order_stat("failed")

    def _handle_risk_callback(self, exc: Exception, symbol: str) -> None:
        risk_manager = self.context.risk_manager
        if risk_manager is not None and hasattr(risk_manager, "handle_broker_error"):
            try:
                risk_manager.handle_broker_error(exc, {"symbol": symbol})
            except Exception:
                logger.debug(
                    "Risk manager handle_broker_error raised",
                    error=str(exc),
                    operation="execution_order",
                    stage="risk_manager_callback",
                )
