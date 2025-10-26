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
from bot_v2.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
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


_UNSET = object()


class _CallableProxy:
    __slots__ = ("_func", "_instance", "return_value", "side_effect")

    def __init__(self, instance: Any, func: Callable[..., Any]) -> None:
        self._instance = instance
        self._func = func
        self.return_value: Any = _UNSET
        self.side_effect: Any = _UNSET

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.side_effect is not _UNSET:
            side_effect = self.side_effect
            if callable(side_effect):
                return side_effect(*args, **kwargs)
            if isinstance(side_effect, BaseException):
                raise side_effect
            return side_effect
        if self.return_value is not _UNSET:
            return self.return_value
        return self._func(self._instance, *args, **kwargs)


class OrderRestMixin(PortfolioRestMixin):
    """High-level order helpers built on top of the Coinbase client."""

    client: CoinbaseClient
    endpoints: CoinbaseEndpoints

    @staticmethod
    def _require_quantity(quantity: Decimal | None, *, context: str) -> Decimal:
        resolved = quantity_from({"quantity": quantity}, default=None)
        if resolved is None:
            raise ValueError(f"{context} requires a quantity")
        return cast(Decimal, resolved)

    class _ListPositionsProxy:
        """Descriptor that allows tests to override list_positions.return_value."""

        def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., Any]:
            if instance is None:
                return self
            func = PortfolioRestMixin.list_positions
            proxy = getattr(instance, "_order_list_positions_proxy", None)
            if (
                proxy is None
                or getattr(proxy, "_func", None) is not func
                or getattr(proxy, "_instance", None) is not instance
            ):
                proxy = _CallableProxy(instance, func)
            setattr(instance, "_order_list_positions_proxy", proxy)
            return proxy

    list_positions = _ListPositionsProxy()  # type: ignore[assignment]

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
            # Enforce client_order_id <= 128 chars per API requirements
            validated_client_id = new_client_id
            if len(validated_client_id) > 128:
                from bot_v2.features.brokerages.coinbase.rest.base import logger

                logger.warning(
                    "new_client_order_id exceeds 128 chars, truncating: %s",
                    validated_client_id,
                    operation="order_edit",
                    stage="client_id_validation",
                )
                validated_client_id = validated_client_id[:128]
            payload["new_client_order_id"] = validated_client_id
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
        build_args = dict(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
        )
        payload = base._build_order_payload(client_id=final_client_id, **build_args)
        try:
            return base._execute_order_payload(symbol, payload, final_client_id)
        except InvalidRequestError as exc:
            if "duplicate" not in str(exc).lower():
                raise
            retry_client_id_root = client_id or "perps_retry"
            retry_client_id = f"{retry_client_id_root}_{uuid.uuid4().hex[:8]}"
            if len(retry_client_id) > 128:
                retry_client_id = retry_client_id[:128]
            logger.info(
                "Retrying order for %s with new client_order_id after duplicate error",
                symbol,
                original_client_id=final_client_id,
                retry_client_id=retry_client_id,
            )
            retry_payload = base._build_order_payload(client_id=retry_client_id, **build_args)
            return base._execute_order_payload(symbol, retry_payload, retry_client_id)

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
        collected: list[dict[str, Any]] = []
        seen_order_ids: set[str] = set()
        cursor: str | None = None
        seen_cursors: set[str] = set()

        def _truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "yes", "1"}
            if isinstance(value, (int, float)):
                return value != 0
            return False

        while True:
            call_kwargs = dict(params)
            if cursor:
                call_kwargs["cursor"] = cursor
            try:
                data = client.list_orders(**call_kwargs) or {}
            except Exception as exc:
                logger.error("Failed to list orders: %s", exc)
                return [to_order(item) for item in collected]

            items = data.get("orders") or data.get("data") or []
            for item in items:
                order_id = str(item.get("order_id") or item.get("id") or "").strip()
                if order_id:
                    if order_id in seen_order_ids:
                        continue
                    seen_order_ids.add(order_id)
                collected.append(item)

            pagination = data.get("pagination") or {}
            next_cursor = (
                data.get("cursor")
                or pagination.get("next_cursor")
                or pagination.get("cursor")
                or pagination.get("next")
            )
            has_more = _truthy(
                data.get("has_next_page")
                or pagination.get("has_next")
                or pagination.get("has_more")
                or pagination.get("has_next_page")
            )

            if not next_cursor or next_cursor in seen_cursors:
                break

            seen_cursors.add(next_cursor)
            cursor = next_cursor

            if not has_more and not items:
                break

        return [to_order(item) for item in collected]

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
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        collected: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        def _truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "yes", "1"}
            if isinstance(value, (int, float)):
                return value != 0
            return False

        while True:
            call_kwargs = dict(params)
            if cursor:
                call_kwargs["cursor"] = cursor
            try:
                data = client.list_fills(**call_kwargs) or {}
            except Exception as exc:
                logger.error("Failed to list fills: %s", exc)
                return collected

            items = data.get("fills") or data.get("data") or []
            collected.extend(items)

            pagination = data.get("pagination") or {}
            next_cursor = (
                data.get("cursor")
                or pagination.get("next_cursor")
                or pagination.get("cursor")
                or pagination.get("next")
            )
            has_more = _truthy(
                data.get("has_next_page")
                or pagination.get("has_next")
                or pagination.get("has_more")
                or pagination.get("has_next_page")
            )

            if not next_cursor or next_cursor in seen_cursors:
                break

            seen_cursors.add(next_cursor)
            cursor = next_cursor

            if not has_more and not items:
                break

        return collected

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


__all__ = ["OrderRestMixin"]
