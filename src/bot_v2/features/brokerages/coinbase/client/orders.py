"""Order management endpoints for Coinbase REST client."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...core.interfaces import BrokerageError, NotFoundError
from ..errors import InvalidRequestError


if TYPE_CHECKING:
    from ._typing import CoinbaseClientProtocol


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
        query = "&".join(f"{key}={value}" for key, value in params.items())
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

    def list_orders_batch(self: CoinbaseClientProtocol, order_ids: list[str]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "list_orders_batch not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path("orders_batch")
        return self._request("GET", path)

    def list_fills(self: CoinbaseClientProtocol, **params: Any) -> dict[str, Any]:
        path = self._get_endpoint_path("fills")
        query = "&".join(f"{key}={value}" for key, value in params.items())
        suffix = f"?{query}" if query else ""
        return self._request("GET", f"{path}{suffix}")


__all__ = ["OrderClientMixin"]
