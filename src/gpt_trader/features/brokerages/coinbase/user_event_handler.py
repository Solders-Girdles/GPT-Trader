"""Coinbase WebSocket user-event handler for order updates and fills."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.coinbase.rest.pnl_service import PnLService
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.ws_events import FillEvent, OrderUpdateEvent
from gpt_trader.persistence.orders_store import OrderRecord, OrdersStore, OrderStatus
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_user_events")

_DEFAULT_DEDUPE_LIMIT = 1000


class CoinbaseUserEventHandler:
    """Handle Coinbase user events and persist order/fill state."""

    def __init__(
        self,
        *,
        broker: Any | None,
        orders_store: OrdersStore | None,
        event_store: Any | None,
        bot_id: str,
        market_data_service: Any | None,
        symbols: list[str] | None,
        dedupe_limit: int = _DEFAULT_DEDUPE_LIMIT,
    ) -> None:
        self._broker = broker
        self._orders_store = orders_store
        self._event_store = event_store
        self._bot_id = bot_id
        self._symbols = symbols or []

        self._pnl_service: PnLService | None = None
        if market_data_service is not None:
            self._pnl_service = PnLService(
                position_store=PositionStateStore(),
                market_data=market_data_service,
            )

        self._recent_fill_keys: deque[str] = deque(maxlen=max(dedupe_limit, 1))
        self._recent_fill_set: set[str] = set()
        self._dedupe_lock = threading.Lock()

    def handle_user_message(self, message: dict[str, Any]) -> None:
        """Parse and handle a raw WebSocket user message."""
        fill = FillEvent.from_message(message)
        if fill is not None:
            self.handle_fill(fill)

        updates = OrderUpdateEvent.from_message(message)
        for update in updates:
            self.handle_order_update(update)

    def handle_order_update(self, event: OrderUpdateEvent) -> None:
        """Handle order status updates from user events."""
        if self._orders_store is None:
            return

        order_id = event.order_id or event.client_order_id
        client_order_id = event.client_order_id or order_id
        if not order_id or not client_order_id:
            return

        timestamp = event.timestamp or datetime.now(timezone.utc)
        status = self._normalize_status(event.status)

        record = OrderRecord(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=event.product_id,
            side=event.side.lower() if event.side else "unknown",
            order_type=event.order_type.lower() if event.order_type else "unknown",
            quantity=event.size,
            price=event.price,
            status=status,
            filled_quantity=event.filled_size,
            average_fill_price=event.avg_price,
            created_at=timestamp,
            updated_at=timestamp,
            bot_id=self._bot_id,
            time_in_force="GTC",
            metadata={"source": "ws_user_event", "event_type": "order_update"},
        )

        try:
            self._orders_store.upsert_by_client_id(record)
        except Exception as exc:
            logger.warning(
                "Failed to persist order update",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="order_update_persist",
                order_id=order_id,
            )

        self._emit_event(
            "user_order_update",
            {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "symbol": event.product_id,
                "status": event.status,
                "filled_size": str(event.filled_size),
                "avg_price": str(event.avg_price) if event.avg_price else None,
            },
        )

    def handle_fill(self, event: FillEvent) -> None:
        """Handle fill events from user channel."""
        if not self._should_process_fill(event):
            return

        fill_payload = {
            "order_id": event.order_id,
            "client_order_id": event.client_order_id,
            "product_id": event.product_id,
            "side": event.side.lower() if event.side else None,
            "price": str(event.fill_price),
            "size": str(event.fill_size),
        }

        fill_delta = self._update_orders_for_fill(event)
        if fill_delta <= 0:
            return

        pnl_payload = dict(fill_payload)
        pnl_payload["size"] = str(fill_delta)

        self._process_fill_for_pnl(pnl_payload)

        self._emit_event(
            "user_fill",
            {
                **fill_payload,
                "fill_delta": str(fill_delta),
                "fee": str(event.fee),
                "commission": str(event.commission),
                "sequence": event.sequence,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            },
        )

    def _process_fill_for_pnl(self, fill: dict[str, Any]) -> None:
        if self._broker is not None and hasattr(self._broker, "process_fill_for_pnl"):
            try:
                self._broker.process_fill_for_pnl(fill)
                return
            except Exception as exc:
                logger.warning(
                    "Broker PnL fill processing failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="pnl_process_fill",
                )

        if self._pnl_service is None:
            return

        try:
            self._pnl_service.process_fill_for_pnl(fill)
        except Exception as exc:
            logger.warning(
                "PnL fill processing failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="pnl_process_fill",
            )

    def _update_orders_for_fill(self, event: FillEvent) -> Decimal:
        if self._orders_store is None:
            return event.fill_size

        order_id = event.order_id or event.client_order_id
        client_order_id = event.client_order_id or order_id
        if not order_id or not client_order_id:
            return Decimal("0")

        existing = None
        try:
            existing = self._orders_store.get_order(order_id)
        except Exception:
            existing = None
        if existing is None and client_order_id and client_order_id != order_id:
            try:
                existing = self._orders_store.get_order(client_order_id)
            except Exception:
                existing = None

        fill_size = event.fill_size
        fill_price = event.fill_price

        if existing is None:
            quantity = fill_size
            filled_quantity = fill_size
            avg_price = fill_price
            fill_delta = fill_size
            status = OrderStatus.PARTIALLY_FILLED
            order_type = "unknown"
            price = None
            time_in_force = "GTC"
            created_at = event.timestamp or datetime.now(timezone.utc)
        else:
            quantity = existing.quantity
            previous_filled = existing.filled_quantity
            fill_delta = fill_size - previous_filled
            if fill_delta <= 0:
                return Decimal("0")
            filled_quantity = fill_size
            avg_price = fill_price if fill_price > 0 else existing.average_fill_price
            status = (
                OrderStatus.FILLED
                if quantity > 0 and filled_quantity >= quantity
                else OrderStatus.PARTIALLY_FILLED
            )
            order_type = existing.order_type
            price = existing.price
            time_in_force = existing.time_in_force
            created_at = existing.created_at

        record = OrderRecord(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=event.product_id,
            side=event.side.lower() if event.side else "unknown",
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=status,
            filled_quantity=filled_quantity,
            average_fill_price=avg_price,
            created_at=created_at,
            updated_at=event.timestamp or datetime.now(timezone.utc),
            bot_id=self._bot_id,
            time_in_force=time_in_force,
            metadata={"source": "ws_user_event", "event_type": "fill"},
        )

        try:
            self._orders_store.upsert_by_client_id(record)
        except Exception as exc:
            logger.warning(
                "Failed to persist fill update",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="fill_persist",
                order_id=order_id,
            )
        return fill_delta

    def _should_process_fill(self, event: FillEvent) -> bool:
        key = self._build_fill_key(
            order_id=event.order_id,
            client_order_id=event.client_order_id,
            product_id=event.product_id,
            price=str(event.fill_price),
            size=str(event.fill_size),
            timestamp=event.timestamp.isoformat() if event.timestamp else None,
            sequence=event.sequence,
        )
        with self._dedupe_lock:
            if key in self._recent_fill_set:
                return False
            self._recent_fill_set.add(key)
            self._recent_fill_keys.append(key)
            while len(self._recent_fill_set) > len(self._recent_fill_keys):
                self._recent_fill_set.discard(self._recent_fill_keys.popleft())
        return True

    def _build_fill_key(
        self,
        *,
        order_id: str | None,
        client_order_id: str | None,
        product_id: str | None,
        price: str | None,
        size: str | None,
        timestamp: str | None,
        sequence: int | None,
    ) -> str:
        return "|".join(
            [
                order_id or "",
                client_order_id or "",
                product_id or "",
                price or "",
                size or "",
                timestamp or "",
                str(sequence or ""),
            ]
        )

    def _normalize_status(self, status: str | None) -> OrderStatus:
        normalized = str(status or "").strip().lower()
        mapping = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "failed": OrderStatus.FAILED,
        }
        return mapping.get(normalized, OrderStatus.OPEN)

    def _merge_average_price(
        self,
        existing_avg: Decimal | None,
        existing_qty: Decimal,
        fill_price: Decimal,
        fill_qty: Decimal,
    ) -> Decimal:
        if existing_avg is None or existing_qty <= 0:
            return fill_price
        total_qty = existing_qty + fill_qty
        if total_qty <= 0:
            return fill_price
        total_cost = existing_avg * existing_qty + fill_price * fill_qty
        return total_cost / total_qty

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_store is None:
            return
        try:
            self._event_store.append(event_type, payload)
        except Exception as exc:
            logger.debug(
                "Failed to emit user event",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="user_event_emit",
                event_type=event_type,
            )
