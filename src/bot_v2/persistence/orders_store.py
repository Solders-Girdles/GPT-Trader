"""
Durable, file-based store for orders and fills to ensure state
can be recovered after restarts.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from bot_v2.features.brokerages.core.interfaces import Order, OrderStatus
from bot_v2.monitoring.system import get_logger
from bot_v2.persistence.json_file_store import JsonFileStore

logger = logging.getLogger(__name__)


@dataclass
class StoredOrder:
    """A simplified, serializable representation of an order."""

    order_id: str
    client_id: str
    symbol: str
    side: str
    order_type: str
    quantity: str
    price: str | None
    status: str
    created_at: str
    updated_at: str
    # Partial fill tracking
    filled_quantity: str | None = None
    avg_fill_price: str | None = None

    @staticmethod
    def from_order(order: Order) -> StoredOrder:
        return StoredOrder(
            order_id=order.id,
            client_id=order.client_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.type.value,
            quantity=str(order.quantity),
            price=str(order.price) if order.price else None,
            status=order.status.value,
            created_at=(
                order.submitted_at.isoformat()
                if order.submitted_at
                else datetime.utcnow().isoformat()
            ),
            updated_at=(
                order.updated_at.isoformat() if order.updated_at else datetime.utcnow().isoformat()
            ),
            avg_fill_price=(
                str(order.avg_fill_price)
                if getattr(order, "avg_fill_price", None) is not None
                else None
            ),
            filled_quantity=(
                str(order.filled_quantity)
                if getattr(order, "filled_quantity", None) is not None
                else None
            ),
        )


class OrdersStore:
    """Manages durable storage of orders to prevent state loss."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.orders_file = storage_path / "orders.jsonl"
        self._orders: dict[str, StoredOrder] = {}
        self._client_id_map: dict[str, str] = {}
        self._lock = threading.RLock()

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._store = JsonFileStore(self.orders_file)
        self._load()

    def _load(self) -> None:
        """Load orders from the JSONL file."""
        for record in self._store.iter_jsonl():
            try:
                data = dict(record)
                legacy_filled_quantity = data.pop("filled_quantity", None)
                if "filled_quantity" not in data and legacy_filled_quantity is not None:
                    data["filled_quantity"] = legacy_filled_quantity
                if "avg_fill_price" not in data:
                    data["avg_fill_price"] = None
                order = StoredOrder(**data)
                self._orders[order.order_id] = order
                self._client_id_map[order.client_id] = order.order_id
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Could not load order record: %s - Error: %s",
                    record,
                    exc,
                )

        logger.info(f"Loaded {len(self._orders)} orders from {self.orders_file}")

    def upsert(self, order: Order) -> None:
        """Update or insert an order."""
        with self._lock:
            prev = self._orders.get(order.id)
            prev_status = prev.status if prev else None
            prev_created_at = prev.created_at if prev else None

            stored_order = StoredOrder.from_order(order)
            self._orders[order.id] = stored_order
            if order.client_id:
                self._client_id_map[order.client_id] = order.id
            # Append to file for durability
            self._store.append_jsonl(asdict(stored_order))

            # Emit status change and round-trip metrics
            try:
                if prev_status != stored_order.status:
                    get_logger().log_order_status_change(
                        order_id=stored_order.order_id,
                        client_order_id=stored_order.client_id,
                        from_status=prev_status,
                        to_status=stored_order.status,
                    )
                # Round-trip when entering FILLED from non-filled
                if stored_order.status.upper() == OrderStatus.FILLED.value.upper() and (
                    not prev_status or prev_status.upper() != OrderStatus.FILLED.value.upper()
                ):
                    try:
                        # Prefer true timestamps from Order object
                        t0 = order.submitted_at
                        t1 = order.updated_at
                        # Fallback to stored strings when missing
                        if not t0 and prev_created_at:
                            t0 = datetime.fromisoformat(prev_created_at)
                        if t0 and t1:
                            rtt_ms = (t1 - t0).total_seconds() * 1000.0
                            get_logger().log_order_round_trip(
                                order_id=stored_order.order_id,
                                client_order_id=stored_order.client_id,
                                round_trip_ms=rtt_ms,
                                submitted_ts=t0.isoformat(),
                                filled_ts=t1.isoformat(),
                            )
                    except Exception as exc:
                        logger.debug(
                            "Failed to log order round trip for %s: %s",
                            stored_order.order_id,
                            exc,
                            exc_info=True,
                        )
            except Exception as exc:
                # Never break on logging but surface for diagnostics
                logger.debug("Order metrics logging failure: %s", exc, exc_info=True)

    def get_by_id(self, order_id: str) -> StoredOrder | None:
        with self._lock:
            return self._orders.get(order_id)

    def get_by_client_id(self, client_id: str) -> StoredOrder | None:
        with self._lock:
            order_id = self._client_id_map.get(client_id)
            return self._orders.get(order_id) if order_id else None

    def get_open_orders(self) -> list[StoredOrder]:
        """Get all orders that are not in a terminal state."""
        terminal_states = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
        with self._lock:
            return [o for o in self._orders.values() if o.status.upper() not in terminal_states]
