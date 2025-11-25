"""
Order management mixin for Coinbase REST service.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.models import to_order
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)


class OrderRestMixin:
    """Mixin for order management operations."""

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> Order:
        """Place a new order."""
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=client_id,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
        )
        return self._execute_order_payload(symbol, payload, client_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            response = self.client.cancel_orders(order_ids=[order_id])
            results = response.get("results", [])
            for res in results:
                if res.get("order_id") == order_id:
                    return res.get("success", False)
            return False
        except Exception:
            return False

    def list_orders(
        self,
        product_id: str | None = None,
        status: list[str] | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """List orders with pagination."""
        orders = []
        cursor = None
        has_more = True

        while has_more:
            try:
                kwargs = {"limit": limit}
                if product_id:
                    kwargs["product_id"] = product_id
                if status:
                    kwargs["order_status"] = status
                if cursor:
                    kwargs["cursor"] = cursor

                response = self.client.list_orders(**kwargs)

                page_orders = response.get("orders", [])
                for item in page_orders:
                    orders.append(to_order(item))

                cursor = response.get("cursor")
                if not cursor or not page_orders:
                    has_more = False
            except Exception:
                has_more = False

        return orders

    def get_order(self, order_id: str) -> Order | None:
        """Get details of a single order."""
        try:
            response = self.client.get_order_historical(order_id)
            order_data = response.get("order")
            if order_data:
                return to_order(order_data)
            return None
        except Exception:
            return None

    def list_fills(
        self,
        product_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List fills with pagination."""
        fills = []
        cursor = None
        has_more = True

        while has_more:
            try:
                kwargs = {"limit": limit}
                if product_id:
                    kwargs["product_id"] = product_id
                if order_id:
                    kwargs["order_id"] = order_id
                if cursor:
                    kwargs["cursor"] = cursor

                response = self.client.list_fills(**kwargs)

                page_fills = response.get("fills", [])
                fills.extend(page_fills)

                cursor = response.get("cursor")
                if not cursor or not page_fills:
                    has_more = False
            except Exception:
                has_more = False

        return fills

    def close_position(
        self,
        symbol: str,
        client_order_id: str | None = None,
        fallback: Callable | None = None,
    ) -> Order:
        """Close position for a symbol."""
        has_pos = False
        if hasattr(self, "list_positions"):
            pos_list = self.list_positions()
            for p in pos_list:
                if p.symbol == symbol and p.quantity > 0:
                    has_pos = True
                    break

        if not has_pos:
            raise ValidationError(f"No open position for {symbol}")

        try:
            payload = {"product_id": symbol}
            if client_order_id:
                payload["client_order_id"] = client_order_id

            response = self.client.close_position(payload)
            return to_order(response.get("order", {}))
        except Exception as e:
            if fallback:
                return fallback()
            raise e
