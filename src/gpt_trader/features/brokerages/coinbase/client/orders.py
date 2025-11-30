"""Order management endpoints for Coinbase REST client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    InvalidRequestError,
    NotFoundError,
)

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.client._typing import CoinbaseClientProtocol


class OrderClientMixin:
    """Order lifecycle helpers built on top of the base REST client."""

    def place_order(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "place_order not available via this client in exchange mode. "
                "Use specialized exchange adapter or set COINBASE_API_MODE=advanced."
            )
        path = self._get_endpoint_path("orders")
        return self._request("POST", path, payload)

    def cancel_orders(self: CoinbaseClientProtocol, order_ids: list[str]) -> dict[str, Any]:
        path = self._get_endpoint_path("orders_batch_cancel")
        return self._request("POST", path, {"order_ids": order_ids})

    def preview_order(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "preview_order not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to call order previews."
            )
        path = self._get_endpoint_path("order_preview")
        return self._request("POST", path, payload)

    def edit_order_preview(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "edit_order_preview not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to edit orders."
            )
        path = self._get_endpoint_path("order_edit_preview")
        return self._request("POST", path, payload)

    def edit_order(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "edit_order not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to edit orders."
            )
        path = self._get_endpoint_path("order_edit")
        return self._request("POST", path, payload)

    def close_position(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "close_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path("close_position")
        return self._request("POST", path, payload)

    def get_order_historical(self: CoinbaseClientProtocol, order_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("order", order_id=order_id)
        return self._request("GET", path)

    def list_orders(self: CoinbaseClientProtocol, **params: Any) -> dict[str, Any]:
        query = urlencode(params) if params else ""
        suffix = f"?{query}" if query else ""

        if self.api_mode == "exchange":
            path_open = self._get_endpoint_path("orders")
            return self._request("GET", f"{path_open}{suffix}")

        path_hist = self._get_endpoint_path("orders_historical")
        try:
            return self._request("GET", f"{path_hist}{suffix}")
        except NotFoundError:
            path_open = self._get_endpoint_path("orders")
            try:
                return self._request("GET", f"{path_open}{suffix}")
            except BrokerageError as exc:
                if "method not allowed" in str(exc).lower():
                    return {"orders": []}
                raise

    def list_orders_batch(
        self: CoinbaseClientProtocol,
        order_ids: Sequence[str],
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve multiple orders in a single call using order IDs."""
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "list_orders_batch not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        resolved_ids = [str(order_id) for order_id in order_ids if order_id]
        if not resolved_ids:
            raise InvalidRequestError("list_orders_batch requires at least one order_id.")
        path = self._get_endpoint_path("orders_batch")
        params: dict[str, Any] = {"order_ids": resolved_ids}
        if cursor:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = str(limit)
        query = urlencode(params, doseq=True)
        final_path = f"{path}?{query}" if query else path
        return self._request("GET", final_path)

    def list_fills(self: CoinbaseClientProtocol, **params: Any) -> dict[str, Any]:
        path = self._get_endpoint_path("fills")
        query = urlencode(params) if params else ""
        suffix = f"?{query}" if query else ""
        return self._request("GET", f"{path}{suffix}")


__all__ = ["OrderClientMixin"]
