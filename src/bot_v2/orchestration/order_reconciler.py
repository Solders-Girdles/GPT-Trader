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
from bot_v2.utilities.telemetry import emit_metric

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
            logger.exception(
                "Failed to load local open orders: %s",
                exc,
                operation="order_reconcile",
                stage="load_local",
            )
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
                            operation="order_reconcile",
                            stage="fetch_exchange_fallback",
                        )
                break
            except Exception as exc:
                logger.exception(
                    "Failed to fetch exchange open orders for status=%s: %s",
                    status,
                    exc,
                    operation="order_reconcile",
                    stage="fetch_exchange",
                    order_status=status.value if hasattr(status, "value") else str(status),
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
        emit_metric(
            self._event_store,
            self._bot_id,
            {
                "event_type": "order_reconcile_snapshot",
                "local_open": len(local_open),
                "exchange_open": len(exchange_open),
            },
            logger=logger,
        )

    async def reconcile_missing_on_exchange(self, diff: OrderDiff) -> None:
        for order_id, local_order in diff.missing_on_exchange.items():
            logger.warning(
                "Order %s is OPEN locally but not on exchange. Fetching final status...",
                order_id,
                operation="order_reconcile",
                stage="missing_on_exchange",
                order_id=order_id,
            )
            final_order: Order | None
            try:
                final_order = await asyncio.to_thread(self._broker.get_order, order_id)
            except Exception as exc:
                logger.debug(
                    "Failed to fetch final status for %s: %s",
                    order_id,
                    exc,
                    operation="order_reconcile",
                    stage="missing_on_exchange",
                    order_id=order_id,
                )
                final_order = None

            if final_order:
                self._persist_exchange_update(final_order)
            else:
                self._assume_cancelled(order_id, local_order)

    def reconcile_missing_locally(self, diff: OrderDiff) -> None:
        for order_id, exchange_order in diff.missing_locally.items():
            logger.warning(
                "Found untracked OPEN order on exchange: %s. Adding to store.",
                order_id,
                operation="order_reconcile",
                stage="missing_locally",
                order_id=order_id,
            )
            try:
                self._orders_store.upsert(exchange_order)
            except Exception as exc:
                logger.debug(
                    "Failed to upsert exchange order %s: %s",
                    order_id,
                    exc,
                    operation="order_reconcile",
                    stage="missing_locally",
                    order_id=order_id,
                )

    async def snapshot_positions(self) -> dict[str, dict[str, str]]:
        try:
            positions = await asyncio.to_thread(self._broker.list_positions)
        except Exception as exc:
            logger.debug(
                "Failed to fetch positions during reconciliation: %s",
                exc,
                operation="order_reconcile",
                stage="positions",
                exc_info=True,
            )
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
            logger.exception(
                "Failed to update orders_store with %s: %s",
                order.id,
                exc,
                operation="order_reconcile",
                stage="persist_order",
                order_id=order.id,
            )
            return

        emit_metric(
            self._event_store,
            self._bot_id,
            {
                "event_type": "order_reconciled",
                "order_id": order.id,
                "status": order.status.value,
            },
            logger=logger,
        )
        logger.info(
            "Updated order %s to status: %s",
            order.id,
            order.status.value,
            operation="order_reconcile",
            stage="persist_order",
            order_id=order.id,
            status=order.status.value,
        )

    def _assume_cancelled(self, order_id: str, local_order: Any) -> None:
        logger.error(
            "Could not retrieve final status for order %s.",
            order_id,
            operation="order_reconcile",
            stage="assume_cancelled",
            order_id=order_id,
        )
        try:
            filled_value = getattr(local_order, "filled_quantity", None)
            quantity = quantity_from(local_order, default=Decimal("0")) or Decimal("0")
            price_raw = getattr(local_order, "price", None)
            price: Decimal | None = None
            if price_raw not in (None, "", "null"):
                try:
                    price = Decimal(str(price_raw))
                except Exception:
                    price = None
            avg_fill_raw = getattr(local_order, "avg_fill_price", None)
            avg_fill: Decimal | None = None
            if avg_fill_raw not in (None, "", "null"):
                try:
                    avg_fill = Decimal(str(avg_fill_raw))
                except Exception:
                    avg_fill = None
            side_token = str(getattr(local_order, "side", "buy")).lower()
            side = OrderSide.BUY if side_token.startswith("b") else OrderSide.SELL
            type_token = str(getattr(local_order, "order_type", "market")).lower()
            try:
                order_type = OrderType(type_token)
            except ValueError:
                order_type = OrderType.MARKET
            submitted_at = self._parse_timestamp(
                getattr(local_order, "created_at", "")
            ) or datetime.now(UTC)
            symbol_value = getattr(local_order, "symbol", "")
            symbol_str = str(symbol_value) if symbol_value else ""
            cancelled_order = Order(
                id=str(getattr(local_order, "order_id", order_id)),
                client_id=getattr(local_order, "client_id", None),
                symbol=symbol_str,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=None,
                tif=TimeInForce.GTC,
                status=OrderStatus.CANCELLED,
                filled_quantity=quantity_from(filled_value, default=None),
                avg_fill_price=avg_fill,
                submitted_at=submitted_at,
                updated_at=datetime.now(UTC),
            )
            self._orders_store.upsert(cancelled_order)
            emit_metric(
                self._event_store,
                self._bot_id,
                {
                    "event_type": "order_reconciled",
                    "order_id": order_id,
                    "status": OrderStatus.CANCELLED.value,
                    "reason": "assumed_cancelled",
                },
                logger=logger,
            )
            logger.info(
                "Marked order %s as cancelled due to missing on exchange",
                order_id,
                operation="order_reconcile",
                stage="assume_cancelled",
                order_id=order_id,
            )
        except Exception as exc:
            logger.debug(
                "Failed to mark %s cancelled during reconciliation: %s",
                order_id,
                exc,
                exc_info=True,
                operation="order_reconcile",
                stage="assume_cancelled",
                order_id=order_id,
            )

    @staticmethod
    def _parse_timestamp(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None
