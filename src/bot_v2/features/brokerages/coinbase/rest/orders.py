"""Order management for Coinbase REST service."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
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

if TYPE_CHECKING:
    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
    from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
    from bot_v2.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase
    from bot_v2.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin


class OrderRestMixin:
    """High-level order helpers built on top of the Coinbase client."""

    client: CoinbaseClient
    endpoints: CoinbaseEndpoints

    @staticmethod
    def _require_quantity(quantity: Decimal | None, *, context: str) -> Decimal:
        resolved = quantity_from({"quantity": quantity}, default=None)
        if resolved is None:
            raise ValueError(f"{context} requires a quantity")
        return cast(Decimal, resolved)

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
        base = cast("CoinbaseRestServiceBase", self)
        payload = base._build_order_payload(
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
        client = cast("CoinbaseClient", base.client)
        response = client.preview_order(payload)
        return cast(dict[str, Any], response or {})

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
        base = cast("CoinbaseRestServiceBase", self)
        payload = base._build_order_payload(
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
        client = cast("CoinbaseClient", base.client)
        response = client.edit_order_preview(payload)
        return cast(dict[str, Any], response or {})

    def edit_order(self, order_id: str, preview_id: str) -> Order:
        payload = {"order_id": order_id, "preview_id": preview_id}
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        data = client.edit_order(payload)
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
        base = cast("CoinbaseRestServiceBase", self)
        payload = base._build_order_payload(
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
        return base._execute_order_payload(symbol, payload, final_client_id)

    def cancel_order(self, order_id: str) -> bool:
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        response = client.cancel_orders([order_id]) or {}
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
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        try:
            data = client.list_orders(**params) or {}
        except Exception as exc:
            logger.error("Failed to list orders: %s", exc)
            return []
        items = data.get("orders") or data.get("data") or []
        return [to_order(item) for item in items]

    def list_orders_batch(
        self,
        order_ids: Sequence[str],
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> list[Order]:
        if not order_ids:
            return []
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        try:
            data = client.list_orders_batch(list(order_ids), cursor=cursor, limit=limit) or {}
        except InvalidRequestError:
            raise
        except Exception as exc:
            logger.error("Failed to list orders batch: %s", exc)
            return []
        items = data.get("orders") or data.get("data") or []
        return [to_order(item) for item in items]

    def get_order(self, order_id: str) -> Order | None:
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        try:
            data = client.get_order_historical(order_id) or {}
            payload = data.get("order") or data
            return to_order(payload)
        except Exception as exc:
            logger.error("Failed to get order %s: %s", order_id, exc)
            return None

    def list_fills(self, symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, str] = {"limit": str(limit)}
        if symbol:
            params["product_id"] = normalize_symbol(symbol)
        try:
            base = cast("CoinbaseRestServiceBase", self)
            client = cast("CoinbaseClient", base.client)
            data = client.list_fills(**params) or {}
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
            list(positions_override)
            if positions_override is not None
            else cast("PortfolioRestMixin", self).list_positions()
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
            base = cast("CoinbaseRestServiceBase", self)
            client = cast("CoinbaseClient", base.client)
            response = client.close_position(payload) or {}
            return to_order(response.get("order") or response)
        except Exception as exc:
            logger.warning("close_position fallback triggered for %s: %s", symbol, exc)
            if fallback is None or requested_quantity is None:
                raise
            return fallback(close_side, requested_quantity, reduce_only)

    def place_scaled_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        price_levels: Sequence[Decimal],
        distribution: str = "linear",
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: bool | None = None,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> list[Order]:
        """Place scaled orders across multiple price levels.

        Distributes the total order quantity across multiple price levels,
        useful for dollar-cost averaging or building positions gradually.

        Args:
            symbol: Trading symbol (e.g., "BTC-PERP")
            side: Order side (BUY or SELL)
            total_quantity: Total quantity to distribute across all levels
            price_levels: List of price levels (must be sorted: low to high for BUY, high to low for SELL)
            distribution: Distribution method:
                - "linear": Equal quantity at each level
                - "weighted": More quantity at better prices (first levels get more)
            tif: Time in force (default: GTC)
            reduce_only: Whether orders can only reduce positions
            leverage: Position leverage for derivatives
            post_only: Post-only flag (maker-only)

        Returns:
            List of placed Order objects

        Raises:
            ValidationError: If price_levels is empty or total_quantity is invalid

        Example:
            # Buy 1 BTC across 5 levels from $45k to $50k
            orders = place_scaled_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                total_quantity=Decimal("1.0"),
                price_levels=[
                    Decimal("45000"),
                    Decimal("46000"),
                    Decimal("47000"),
                    Decimal("48000"),
                    Decimal("50000"),
                ],
                distribution="weighted",  # More quantity at lower prices
            )

        Note:
            Per Oct 2025 changelog, scaled orders are now officially supported
            by Coinbase Advanced Trade API. This implementation provides a
            client-side distribution strategy.
        """
        if not price_levels:
            raise ValidationError("price_levels cannot be empty", field="price_levels")

        if total_quantity <= 0:
            raise ValidationError(
                f"total_quantity must be positive, got {total_quantity}", field="total_quantity"
            )

        num_levels = len(price_levels)

        # Calculate quantity distribution
        if distribution == "linear":
            # Equal distribution across all levels
            quantities = [total_quantity / Decimal(str(num_levels))] * num_levels
        elif distribution == "weighted":
            # Weighted distribution: more quantity at better prices
            # Use triangular distribution: weights = [n, n-1, n-2, ..., 1]
            weights = [Decimal(str(num_levels - i)) for i in range(num_levels)]
            total_weight = sum(weights)
            quantities = [(w / total_weight) * total_quantity for w in weights]
        else:
            raise ValidationError(
                f"Unsupported distribution method: {distribution}. Use 'linear' or 'weighted'.",
                field="distribution",
                value=distribution,
            )

        # Place orders at each level
        orders: list[Order] = []
        for price, quantity in zip(price_levels, quantities):
            try:
                order = self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=quantity,
                    price=price,
                    tif=tif,
                    reduce_only=reduce_only,
                    leverage=leverage,
                    post_only=post_only,
                )
                orders.append(order)
                logger.info(
                    "Scaled order placed: %s %s @ %s (qty: %s)",
                    side.value,
                    symbol,
                    price,
                    quantity,
                )
            except Exception as exc:
                logger.error(
                    "Failed to place scaled order at price %s: %s. Placed %d/%d orders.",
                    price,
                    exc,
                    len(orders),
                    num_levels,
                )
                # Return partial results if some orders succeeded
                if orders:
                    logger.warning(
                        "Partial scaled order execution: %d/%d orders placed", len(orders), num_levels
                    )
                    return orders
                # Re-raise if no orders succeeded
                raise

        logger.info("Scaled order complete: %d orders placed for %s", len(orders), symbol)
        return orders


__all__ = ["OrderRestMixin"]
