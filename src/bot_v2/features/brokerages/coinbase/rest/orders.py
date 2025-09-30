"""Order management for Coinbase REST service."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.coinbase.models import normalize_symbol, to_order
from bot_v2.features.brokerages.coinbase.rest.base import logger
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from bot_v2.utilities.quantities import quantity_from


class OrderRestMixin:
    """High-level order helpers built on top of the Coinbase client."""

    @staticmethod
    def _require_quantity(quantity: Decimal | None, *, context: str) -> Decimal:
        resolved = quantity_from({"quantity": quantity}, default=None)
        if resolved is None:
            raise ValueError(f"{context} requires a quantity")
        return resolved

    def preview_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: bool | None = None,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> dict[str, Any]:
        order_quantity = self._require_quantity(quantity, context="preview_order")
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=None,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
            include_client_id=False,
        )
        return self.client.preview_order(payload)  # type: ignore[attr-defined]

    def edit_order_preview(
        self,
        *,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        new_client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> dict[str, Any]:
        order_quantity = self._require_quantity(quantity, context="edit_order_preview")
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=None,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
            include_client_id=False,
        )
        payload["order_id"] = order_id
        if new_client_id:
            payload["new_client_order_id"] = new_client_id
        return self.client.edit_order_preview(payload)  # type: ignore[attr-defined]

    def edit_order(self, order_id: str, preview_id: str) -> Order:
        payload = {"order_id": order_id, "preview_id": preview_id}
        data = self.client.edit_order(payload)  # type: ignore[attr-defined]
        return to_order(data or {})

    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> Order:
        order_quantity = self._require_quantity(quantity, context="place_order")
        final_client_id = client_id or f"perps_{uuid.uuid4().hex[:12]}"
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=final_client_id,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
        )
        return self._execute_order_payload(symbol, payload, final_client_id)

    def cancel_order(self, order_id: str) -> bool:
        response = self.client.cancel_orders([order_id]) or {}
        results = response.get("results") or response.get("data") or []
        for entry in results:
            if str(entry.get("order_id")) == order_id and entry.get("success") is True:
                return True
        cancelled = response.get("cancelled_order_ids") or []
        return order_id in cancelled

    def list_orders(
        self,
        status: OrderStatus | str | None = None,
        symbol: str | None = None,
    ) -> list[Order]:
        params: dict[str, str] = {}
        if status:
            params["order_status"] = (
                status.value if isinstance(status, OrderStatus) else str(status)
            )
        if symbol:
            params["product_id"] = normalize_symbol(symbol)
        try:
            if hasattr(self.client, "list_orders"):
                data = self.client.list_orders(**params) or {}
            else:  # pragma: no cover - legacy path
                data = self.client.get(self.endpoints.list_orders(), params=params)
        except Exception as exc:
            logger.error("Failed to list orders: %s", exc)
            return []
        items = data.get("orders") or data.get("data") or []
        return [to_order(item) for item in items]

    def get_order(self, order_id: str) -> Order | None:
        try:
            if hasattr(self.client, "get_order_historical"):
                data = self.client.get_order_historical(order_id) or {}
                payload = data.get("order") or data
            else:  # pragma: no cover - legacy path
                payload = self.client.get(self.endpoints.get_order(order_id)) or {}
            return to_order(payload)
        except Exception as exc:
            logger.error("Failed to get order %s: %s", order_id, exc)
            return None

    def list_fills(self, symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, str] = {"limit": str(limit)}
        if symbol:
            params["product_id"] = normalize_symbol(symbol)
        try:
            data = self.client.list_fills(**params) or {}
        except Exception as exc:
            logger.error("Failed to list fills: %s", exc)
            return []
        return data.get("fills") or data.get("data") or []

    def close_position(
        self,
        symbol: str,
        quantity: Decimal | None = None,
        reduce_only: bool = True,
        positions_override: Sequence[Position] | None = None,
        fallback: Callable[[OrderSide, Decimal, bool], Order] | None = None,
    ) -> Order:
        product_id = normalize_symbol(symbol)
        requested_quantity = quantity_from({"quantity": quantity}, default=None)
        if requested_quantity is not None:
            requested_quantity = abs(requested_quantity)
        close_side: OrderSide | None = None

        positions = (
            list(positions_override) if positions_override is not None else self.list_positions()
        )

        if requested_quantity is None or close_side is None:
            current = next((p for p in positions if p.symbol == product_id), None)
            if not current:
                raise ValidationError(f"No open position for {symbol}")
            current_quantity = quantity_from(current, default=Decimal("0")) or Decimal("0")
            if current_quantity == 0:
                raise ValidationError(f"Position already flat for {symbol}")
            requested_quantity = abs(current_quantity)
            close_side = OrderSide.SELL if current_quantity > 0 else OrderSide.BUY
        else:
            close_side = (
                OrderSide.SELL if requested_quantity and requested_quantity > 0 else OrderSide.BUY
            )

        payload: dict[str, Any] = {
            "product_id": product_id,
            "reduce_only": reduce_only,
            "size": str(requested_quantity),
            "quantity": str(requested_quantity),
            "side": close_side.value.upper(),
        }

        try:
            response = self.client.close_position(payload) or {}
            return to_order(response.get("order") or response)
        except Exception as exc:
            logger.warning("close_position fallback triggered for %s: %s", symbol, exc)
            if fallback is None or requested_quantity is None:
                raise
            return fallback(close_side, requested_quantity, reduce_only)


__all__ = ["OrderRestMixin"]
