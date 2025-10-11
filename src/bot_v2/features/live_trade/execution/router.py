"""Order routing helpers (client ids, broker submission, replace)."""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections.abc import MutableMapping
from decimal import Decimal
from typing import Any, cast

from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)


class OrderRouter:
    """Manage client ids, broker submission, and cancel/replace flows."""

    def __init__(
        self,
        *,
        broker: Any,
        pending_orders: MutableMapping[str, Order],
        client_order_map: MutableMapping[str, str],
        order_metrics: MutableMapping[str, int],
    ) -> None:
        self._broker = broker
        self._pending = pending_orders
        self._client_map = client_order_map
        self._metrics = order_metrics

    # ------------------------------------------------------------------
    # Client identifiers
    # ------------------------------------------------------------------
    def prepare_client_id(self, client_id: str | None, symbol: str, side: OrderSide) -> str:
        return (
            client_id or f"{symbol}_{side.value}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )

    def check_duplicate(self, client_id: str) -> Order | None:
        if client_id in self._client_map:
            logger.warning("Duplicate client_id %s, returning existing order", client_id)
            existing_id = self._client_map[client_id]
            return self._pending.get(existing_id)
        return None

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
        time_in_force: TimeInForce,
        client_id: str,
        reduce_only: bool,
        leverage: int | None,
    ) -> Order | None:
        broker_place = getattr(self._broker, "place_order")
        params = inspect.signature(broker_place).parameters

        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": order_quantity,
            "client_id": client_id,
            "reduce_only": reduce_only,
            "leverage": leverage,
        }

        if "limit_price" in params:
            kwargs["limit_price"] = limit_price
        elif "price" in params:
            kwargs["price"] = limit_price

        if "stop_price" in params:
            kwargs["stop_price"] = stop_price

        if isinstance(time_in_force, TimeInForce):
            tif_value_enum = time_in_force
            tif_value_str = time_in_force.value
        else:  # pragma: no cover - defensive (string inputs)
            try:
                tif_value_enum = TimeInForce[str(time_in_force).upper()]
            except Exception:
                tif_value_enum = TimeInForce.GTC
            tif_value_str = tif_value_enum.value

        if "time_in_force" in params:
            kwargs["time_in_force"] = tif_value_str
        if "tif" in params:
            kwargs["tif"] = tif_value_enum

        order = cast(Order | None, broker_place(**kwargs))

        if order:
            self._pending[order.id] = order
            self._client_map[client_id] = order.id
            self._metrics["placed"] += 1
            logger.info("Placed order %s: %s %s %s", order.id, side.value, order_quantity, symbol)

        return order

    # ------------------------------------------------------------------
    # Cancel/replace
    # ------------------------------------------------------------------
    def cancel_and_replace(
        self,
        *,
        order_id: str,
        new_price: Decimal | None,
        new_size: Decimal | None,
        max_retries: int,
    ) -> Order | None:
        original = self._pending.get(order_id)
        if not original:
            logger.error("Order %s not found for cancel/replace", order_id)
            return None

        replace_client_id = f"{original.client_id}_replace_{int(time.time() * 1000)}"

        for attempt in range(max_retries):
            try:
                if bool(self._broker.cancel_order(order_id)):
                    self._metrics["cancelled"] += 1
                    del self._pending[order_id]
                    break
            except Exception as exc:
                logger.warning("Cancel attempt %s failed: %s", attempt + 1, exc, exc_info=True)
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.5 * (2**attempt))  # Exponential backoff

        original_quantity = original.quantity
        replacement_side = OrderSide.SELL if original.side == OrderSide.BUY else OrderSide.BUY
        replacement_type = original.type
        replacement_tif = original.tif

        new_quantity = new_size if new_size is not None else original_quantity
        new_quantity = (
            new_quantity if isinstance(new_quantity, Decimal) else Decimal(str(new_quantity))
        )

        new_price_decimal = Decimal(str(new_price)) if new_price is not None else None

        replacement_limit = (
            new_price_decimal
            if replacement_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
            else original.price
        )
        replacement_stop = (
            new_price_decimal
            if replacement_type in (OrderType.STOP, OrderType.STOP_LIMIT)
            else original.stop_price
        )

        return self.submit(
            symbol=original.symbol,
            side=replacement_side,
            order_type=replacement_type,
            order_quantity=new_quantity,
            limit_price=replacement_limit,
            stop_price=replacement_stop,
            time_in_force=replacement_tif,
            client_id=replace_client_id,
            reduce_only=False,
            leverage=None,
        )
