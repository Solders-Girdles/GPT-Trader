"""Order reconciliation helpers for live trading bots."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore
from bot_v2.utilities.quantities import quantity_from

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderDiff:
    """High level diff between local and exchange open orders."""

    missing_on_exchange: dict[str, Any]
    missing_locally: dict[str, Order]


class OrderReconciler:
    """Coordinates startup reconciliation between local storage and the exchange."""

    _INTERESTED_STATUSES = (
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
    )

    def __init__(
        self,
        *,
        broker: Any,
        orders_store: OrdersStore,
        event_store: EventStore,
        bot_id: str,
    ) -> None:
        self._broker = broker
        self._orders_store = orders_store
        self._event_store = event_store
        self._bot_id = bot_id

    # ------------------------------------------------------------------
    def fetch_local_open_orders(self) -> dict[str, Any]:
        try:
            return {order.order_id: order for order in self._orders_store.get_open_orders()}
        except Exception as exc:
            logger.exception("Failed to load local open orders: %s", exc, exc_info=True)
            return {}

    async def fetch_exchange_open_orders(self) -> dict[str, Order]:
        exchange_open: dict[str, Order] = {}
        for status in self._INTERESTED_STATUSES:
            try:
                orders = await asyncio.to_thread(self._broker.list_orders, status=status)
                for order in orders or []:
                    exchange_open[order.id] = order
            except TypeError:
                if not exchange_open:
                    try:
                        orders = await asyncio.to_thread(self._broker.list_orders)
                        for order in orders or []:
                            if order.status in self._INTERESTED_STATUSES:
                                exchange_open[order.id] = order
                    except Exception as exc:
                        logger.exception(
                            "Failed fallback list_orders() during reconciliation: %s",
                            exc,
                            exc_info=True,
                        )
                break
            except Exception as exc:
                logger.exception(
                    "Failed to fetch exchange open orders for status=%s: %s",
                    status,
                    exc,
                    exc_info=True,
                )
        return exchange_open

    @staticmethod
    def diff_orders(local_open: dict[str, Any], exchange_open: dict[str, Order]) -> OrderDiff:
        missing_on_exchange = {
            order_id: local_open[order_id]
            for order_id in local_open
            if order_id not in exchange_open
        }
        missing_locally = {
            order_id: exchange_open[order_id]
            for order_id in exchange_open
            if order_id not in local_open
        }
        return OrderDiff(missing_on_exchange=missing_on_exchange, missing_locally=missing_locally)

    async def record_snapshot(
        self, local_open: dict[str, Any], exchange_open: dict[str, Order]
    ) -> None:
        try:
            self._event_store.append_metric(
                bot_id=self._bot_id,
                metrics={
                    "event_type": "order_reconcile_snapshot",
                    "local_open": len(local_open),
                    "exchange_open": len(exchange_open),
                },
            )
        except Exception as exc:
            logger.exception("Failed to persist order reconciliation snapshot: %s", exc)

    async def reconcile_missing_on_exchange(self, diff: OrderDiff) -> None:
        for order_id, local_order in diff.missing_on_exchange.items():
            logger.warning(
                "Order %s is OPEN locally but not on exchange. Fetching final status...",
                order_id,
            )
            final_order: Order | None
            try:
                final_order = await asyncio.to_thread(self._broker.get_order, order_id)
            except Exception as exc:
                logger.debug("Failed to fetch final status for %s: %s", order_id, exc)
                final_order = None

            if final_order:
                self._persist_exchange_update(final_order)
            else:
                self._assume_cancelled(order_id, local_order)

    def reconcile_missing_locally(self, diff: OrderDiff) -> None:
        for order_id, exchange_order in diff.missing_locally.items():
            logger.warning("Found untracked OPEN order on exchange: %s. Adding to store.", order_id)
            try:
                self._orders_store.upsert(exchange_order)
            except Exception as exc:
                logger.debug("Failed to upsert exchange order %s: %s", order_id, exc)

    async def snapshot_positions(self) -> dict[str, dict[str, str]]:
        try:
            positions = await asyncio.to_thread(self._broker.list_positions)
        except Exception as exc:
            logger.debug("Failed to fetch positions during reconciliation: %s", exc, exc_info=True)
            return {}

        snapshot: dict[str, dict[str, str]] = {}
        for pos in positions or []:
            symbol = getattr(pos, "symbol", None)
            if not symbol:
                continue
            quantity_val = quantity_from(pos, default=None)
            if quantity_val is None:
                continue
            snapshot[symbol] = {
                "quantity": str(quantity_val),
                "side": str(getattr(pos, "side", "")),
            }
        return snapshot

    # ------------------------------------------------------------------
    def _persist_exchange_update(self, order: Order) -> None:
        try:
            self._orders_store.upsert(order)
        except Exception as exc:
            logger.exception("Failed to update orders_store with %s: %s", order.id, exc)
            return

        try:
            self._event_store.append_metric(
                bot_id=self._bot_id,
                metrics={
                    "event_type": "order_reconciled",
                    "order_id": order.id,
                    "status": order.status.value,
                },
            )
        except Exception as exc:
            logger.exception("Failed to log order reconciliation for %s: %s", order.id, exc)
        logger.info("Updated order %s to status: %s", order.id, order.status.value)

    def _assume_cancelled(self, order_id: str, local_order: Any) -> None:
        logger.error("Could not retrieve final status for order %s.", order_id)
        try:
            filled_value = getattr(
                local_order, "filled_quantity", getattr(local_order, "filled_quantity", None)
            )
            cancelled_order = Order(
                id=getattr(local_order, "order_id", order_id),
                client_id=getattr(local_order, "client_id", None),
                symbol=getattr(local_order, "symbol", ""),
                side=OrderSide(str(getattr(local_order, "side", "buy")).lower()),
                type=OrderType(str(getattr(local_order, "order_type", "market")).lower()),
                quantity=quantity_from(local_order),
                price=(
                    Decimal(str(getattr(local_order, "price", "")))
                    if getattr(local_order, "price", None)
                    else None
                ),
                stop_price=None,
                tif=TimeInForce.GTC,
                status=OrderStatus.CANCELLED,
                filled_quantity=quantity_from(filled_value),
                avg_fill_price=(
                    Decimal(str(getattr(local_order, "avg_fill_price", "")))
                    if getattr(local_order, "avg_fill_price", None)
                    else None
                ),
                submitted_at=self._parse_timestamp(getattr(local_order, "created_at", "")),
                updated_at=datetime.now(UTC),
            )
            self._orders_store.upsert(cancelled_order)
            try:
                self._event_store.append_metric(
                    bot_id=self._bot_id,
                    metrics={
                        "event_type": "order_reconciled",
                        "order_id": order_id,
                        "status": OrderStatus.CANCELLED.value,
                        "reason": "assumed_cancelled",
                    },
                )
            except Exception as exc:
                logger.exception("Failed to log assumed cancellation for %s: %s", order_id, exc)
            logger.info("Marked order %s as cancelled due to missing on exchange", order_id)
        except Exception as exc:
            logger.debug(
                "Failed to mark %s cancelled during reconciliation: %s",
                order_id,
                exc,
                exc_info=True,
            )

    @staticmethod
    def _parse_timestamp(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None
