"""
Order submission and recording for live trading execution.

This module handles the actual submission of orders to the broker and
recording of order events, previews, and rejections.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.core.interfaces import (
    IBrokerage,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.monitoring.system import LogLevel
from gpt_trader.monitoring.system import get_logger as get_monitoring_logger
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="order_submission")


class OrderSubmitter:
    """Handles order submission and event recording."""

    def __init__(
        self,
        broker: IBrokerage,
        event_store: EventStore,
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
        """
        self.broker = broker
        self.event_store = event_store
        self.bot_id = bot_id
        self.open_orders = open_orders
        self.integration_mode = integration_mode

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
            self.event_store,
            self.bot_id,
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
                component="LiveExecutionEngine",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
            )
        except Exception:
            pass

    def record_rejection(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal | None, reason: str
    ) -> None:
        """Record order rejection for analysis."""
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
            operation="order_rejected",
            stage="record",
        )
        # Persist an order_rejected metric for downstream analysis/tests
        emit_metric(
            self.event_store,
            self.bot_id,
            {
                "event_type": "order_rejected",
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "price": str(price) if price is not None else "market",
                "reason": reason,
            },
            logger=get_monitoring_logger(),
        )
        try:
            get_monitoring_logger().log_order_status_change(
                order_id="",
                client_order_id="",
                from_status=None,
                to_status="REJECTED",
                reason=reason,
            )
        except Exception:
            pass

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
        submit_id = (
            client_order_id or f"{self.bot_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        )
        if self.integration_mode:
            forced_id = os.getenv("INTEGRATION_TEST_ORDER_ID")
            if forced_id:
                submit_id = forced_id
        try:
            get_monitoring_logger().log_order_submission(
                client_order_id=submit_id,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=float(order_quantity),
                price=float(price) if price is not None else None,
            )
        except Exception:
            pass

        try:
            order = self.broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=order_quantity,
                price=price,
                stop_price=stop_price,
                tif=tif if tif is not None else None,
                reduce_only=reduce_only,
                leverage=leverage,
                client_id=submit_id,
            )
        except TypeError as exc:
            if not self.integration_mode or "unexpected keyword argument" not in str(exc):
                raise
            order = self._invoke_integration_place_order(
                submit_id,
                symbol,
                side,
                order_type,
                order_quantity,
                price,
                stop_price,
                tif,
            )
        else:
            if inspect.isawaitable(order):
                if not self.integration_mode:
                    raise TypeError("Broker place_order returned awaitable in non-integration mode")
                order = self._await_integration_call(order)

        if order and order.id:
            status_value = getattr(order, "status", None)
            status_name = (
                status_value.value if hasattr(status_value, "value") else str(status_value or "")
            )
            if str(status_name).upper() in {"REJECTED", "CANCELLED", "FAILED"}:
                if self.integration_mode:
                    self.event_store.store_event(
                        "order_rejected",
                        {
                            "order_id": order.id,
                            "symbol": symbol,
                            "status": str(status_name),
                        },
                    )
                    return order
                self.record_rejection(
                    symbol,
                    side.value,
                    order_quantity,
                    price if price is not None else effective_price,
                    f"broker_status_{status_name}",
                )
                try:
                    self.event_store.append_error(
                        bot_id=self.bot_id,
                        message="broker_order_rejected",
                        context={
                            "symbol": symbol,
                            "status": str(status_name),
                            "quantity": str(order_quantity),
                        },
                    )
                except Exception:
                    pass
                if not self.integration_mode:
                    raise RuntimeError(f"Order rejected by broker: {status_name}")
                return order
            self.open_orders.append(order.id)
            display_price = price if price is not None else "market"
            logger.info(
                "Order placed: %s %s %s @ %s (reduce_only=%s)",
                side.value,
                order_quantity,
                symbol,
                display_price,
                reduce_only,
                order_id=str(order.id),
                symbol=symbol,
                side=side.value,
                quantity=float(order_quantity),
                price=float(display_price) if isinstance(display_price, Decimal) else display_price,
                reduce_only=reduce_only,
                operation="order_submit",
                stage="placed",
            )
            logger.info(
                "Trade recorded: %s %s %s @ %s (reduce_only=%s)",
                side.value,
                order_quantity,
                symbol,
                display_price,
                reduce_only,
                order_id=str(order.id),
                symbol=symbol,
                side=side.value,
                quantity=float(order_quantity),
                price=float(display_price) if isinstance(display_price, Decimal) else display_price,
                reduce_only=reduce_only,
                operation="order_submit",
                stage="trade_record",
            )
            try:
                get_monitoring_logger().log_order_status_change(
                    order_id=str(order.id),
                    client_order_id=getattr(order, "client_order_id", submit_id),
                    from_status=None,
                    to_status=getattr(order, "status", "SUBMITTED"),
                )
            except Exception:
                pass
            try:
                trade_quantity = getattr(order, "quantity", order_quantity)
                trade_payload = {
                    "order_id": order.id,
                    "client_order_id": getattr(order, "client_order_id", submit_id),
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(trade_quantity),
                    "price": str(order.price or price or effective_price or "market"),
                    "status": getattr(order, "status", "SUBMITTED"),
                }
                self.event_store.append_trade(self.bot_id, trade_payload)
            except Exception:
                pass
            return order if self.integration_mode else order.id

        logger.error(
            "Order placement failed",
            symbol=symbol,
            side=side.value,
            quantity=float(order_quantity),
            operation="order_submit",
            stage="failed",
        )
        try:
            self.event_store.append_error(
                bot_id=self.bot_id,
                message="order_placement_failed",
                context={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(order_quantity),
                },
            )
        except Exception:
            pass
        return None

    def _invoke_integration_place_order(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce | None,
    ) -> Any:
        tif_value = tif if isinstance(tif, TimeInForce) else TimeInForce.GTC
        now = datetime.utcnow()
        order_obj = Order(
            id=submit_id,
            client_id=submit_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif_value,
            status=OrderStatus.PENDING,
            submitted_at=now,
            updated_at=now,
        )
        result = self.broker.place_order(order_obj)
        if inspect.isawaitable(result):
            return self._await_integration_call(result)
        return result

    @staticmethod
    def _await_integration_call(coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
