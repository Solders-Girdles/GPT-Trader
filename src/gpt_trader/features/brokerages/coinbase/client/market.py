"""Market data endpoints for Coinbase REST client."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.client._typing import CoinbaseClientProtocol


class MarketDataClientMixin:
    """Methods for market product discovery and quotes."""

    def get_products(
        self: CoinbaseClientProtocol,
        *,
        product_type: str | None = None,
        contract_expiry_type: str | None = None,
    ) -> dict[str, Any]:
        """Fetch products from Coinbase API.

        Args:
            product_type: Filter by product type ("spot", "future")
            contract_expiry_type: Filter by contract expiry ("perpetual", "expiring")

        Returns:
            API response with products list

        Note:
            Per Oct 2025 changelog: explicit filtering recommended to avoid
            implicit behavior when expiry parameters are present.
        """
        path = self._get_endpoint_path("products")
        params: dict[str, str] = {}
        if product_type:
            params["product_type"] = product_type
        if contract_expiry_type:
            params["contract_expiry_type"] = contract_expiry_type
        if params:
            path = self._build_path_with_params(path, params)
        return self._request("GET", path)

    def get_ticker(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("ticker", product_id=product_id)
        return self._request("GET", path)

    def get_candles(
        self: CoinbaseClientProtocol,
        product_id: str,
        granularity: str,
        limit: int = 300,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        path = self._get_endpoint_path("candles", product_id=product_id)
        params: dict[str, Any] = {
            "granularity": granularity,
            "limit": limit,
        }

        def _format(dt: datetime) -> str:
            ts = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
            return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")

        if start is not None:
            params["start"] = _format(start)
        if end is not None:
            params["end"] = _format(end)

        final_path = self._build_path_with_params(path, params)
        return self._request("GET", final_path)

    def get_product(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("product", product_id=product_id)
        return self._request("GET", path)

    def get_product_book(
        self: CoinbaseClientProtocol, product_id: str, level: int = 2
    ) -> dict[str, Any]:
        path = self._get_endpoint_path("order_book", product_id=product_id)
        if self.api_mode == "exchange":
            path = f"{path}?level={level}"
        else:
            path = f"{path}?product_id={product_id}&level={level}"
        return self._request("GET", path)

    def get_market_products(self: CoinbaseClientProtocol) -> dict[str, Any]:
        if self.api_mode != "advanced":
            raise InvalidRequestError(
                "get_market_products requires advanced mode (COINBASE_API_MODE=advanced)."
            )
        return self._request("GET", self._get_endpoint_path("public_products"))

    def get_market_product(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        if self.api_mode != "advanced":
            raise InvalidRequestError(
                "get_market_product requires advanced mode (COINBASE_API_MODE=advanced)."
            )
        return self._request(
            "GET", self._get_endpoint_path("public_product", product_id=product_id)
        )

    def get_market_product_ticker(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("public_ticker", product_id=product_id)
        return self._request("GET", path)

    def get_market_product_candles(
        self: CoinbaseClientProtocol, product_id: str, granularity: str, limit: int = 200
    ) -> dict[str, Any]:
        path = self._get_endpoint_path("public_candles", product_id=product_id)
        return self._request("GET", f"{path}?granularity={granularity}&limit={limit}")

    def get_market_product_book(
        self: CoinbaseClientProtocol, product_id: str, level: int = 2
    ) -> dict[str, Any]:
        return self.get_product_book(product_id, level)

    def get_best_bid_ask(self: CoinbaseClientProtocol, product_ids: list[str]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "get_best_bid_ask not available in exchange mode. "
                "Use get_ticker for individual products instead. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path("best_bid_ask")
        query = ",".join(product_ids)
        return self._request("GET", f"{path}?product_ids={query}")

    def list_products(
        self: CoinbaseClientProtocol,
        product_type: str | None = None,
        contract_expiry_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return available products, normalising legacy payload shapes.

        Args:
            product_type: Filter by product type ("spot", "future")
            contract_expiry_type: Filter by contract expiry ("perpetual", "expiring")

        Returns:
            List of product dictionaries

        Note:
            Per Oct 2025 changelog: explicit filtering recommended.
            Filters are applied at API level when possible (via get_products),
            with client-side fallback for additional filtering.
        """
        payload = self.get_products(
            product_type=product_type,
            contract_expiry_type=contract_expiry_type,
        )
        products: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            raw_products = payload.get("products")
            if isinstance(raw_products, list):
                products = [p for p in raw_products if isinstance(p, dict)]
        elif isinstance(payload, list):
            products = [p for p in payload if isinstance(p, dict)]

        # Client-side fallback filtering if API didn't filter
        if product_type:
            target = product_type.upper()
            products = [
                product
                for product in products
                if (product.get("product_type") or "").upper() == target
            ]

        if contract_expiry_type:
            target = contract_expiry_type.upper()
            products = [
                product
                for product in products
                if (product.get("contract_expiry_type") or "").upper() == target
            ]

        return products


__all__ = ["MarketDataClientMixin"]
