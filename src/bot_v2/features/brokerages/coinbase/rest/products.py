"""Product discovery helpers for Coinbase REST service."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.coinbase.models import (
    normalize_symbol,
    to_candle,
    to_product,
    to_quote,
)
from bot_v2.features.brokerages.coinbase.rest.base import logger
from bot_v2.features.brokerages.core.interfaces import MarketType, Product, Quote

if TYPE_CHECKING:
    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
    from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
    from bot_v2.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog


class ProductRestMixin:
    """Product discovery and market data helpers."""

    client: CoinbaseClient
    endpoints: CoinbaseEndpoints
    product_catalog: ProductCatalog

    def list_products(
        self,
        market: MarketType | None = None,
        *,
        product_type: str | None = None,
        contract_expiry_type: str | None = None,
    ) -> list[Product]:
        """List tradable products with optional filtering.

        Args:
            market: Filter by MarketType enum (legacy parameter for backward compatibility)
            product_type: Filter by product type ("spot", "future")
            contract_expiry_type: Filter by contract expiry ("perpetual", "expiring")

        Returns:
            List of Product objects matching the filters

        Note:
            Per Oct 2025 changelog: explicit filtering recommended to avoid
            implicit behavior when expiry parameters are present. Use product_type
            and contract_expiry_type for explicit API-level filtering.

        Examples:
            # List all perpetual futures
            products = list_products(product_type="future", contract_expiry_type="perpetual")

            # List all spot products
            products = list_products(product_type="spot")

            # Legacy usage (still supported)
            products = list_products(market=MarketType.PERPETUAL)
        """
        try:
            base = cast("CoinbaseRestServiceBase", self)
            client = cast("CoinbaseClient", base.client)
            response = client.get_products(
                product_type=product_type,
                contract_expiry_type=contract_expiry_type,
            ) or {}
        except Exception as exc:
            logger.error("Failed to list products: %s", exc)
            return []

        items = response.get("products") or response.get("data") or []
        products: list[Product] = []
        for item in items:
            product = to_product(item)
            if product.market_type == MarketType.PERPETUAL:
                product = self._enrich_with_funding(product)
            if market is None or product.market_type == market:
                products.append(product)

        if market in (None, MarketType.PERPETUAL):
            perps = [p for p in products if p.market_type == MarketType.PERPETUAL]
            logger.info("Found %d perpetual products", len(perps))
        return products

    def get_product(self, symbol: str) -> Product | None:
        normalized = normalize_symbol(symbol)
        product: Product | None = None
        try:
            base = cast("CoinbaseRestServiceBase", self)
            catalog = cast("ProductCatalog", base.product_catalog)
            client = cast("CoinbaseClient", base.client)
            product = catalog.get(client, normalized)
        except Exception as exc:
            logger.debug("Product catalog lookup failed for %s: %s", normalized, exc, exc_info=True)
            product = None

        if product is None:
            try:
                base = cast("CoinbaseRestServiceBase", self)
                client = cast("CoinbaseClient", base.client)
                if hasattr(client, "get_product"):
                    data = client.get_product(normalized) or {}
                else:  # pragma: no cover - legacy endpoint
                    data = client.get(base.endpoints.get_product(normalized)) or {}
                product = to_product(data)
            except Exception as exc:
                logger.error("Failed to fetch product %s: %s", symbol, exc)
                return None

        if product.market_type == MarketType.PERPETUAL:
            product = self._enrich_with_funding(product)
        return product

    def _enrich_with_funding(self, product: Product) -> Product:
        if not self.endpoints.supports_derivatives():
            return product
        try:
            base = cast("CoinbaseRestServiceBase", self)
            catalog = cast("ProductCatalog", base.product_catalog)
            client = cast("CoinbaseClient", base.client)
            funding_rate, next_funding_time = catalog.get_funding(client, product.symbol)
            if funding_rate is not None:
                product.funding_rate = Decimal(str(funding_rate))
            if next_funding_time is not None:
                product.next_funding_time = next_funding_time
        except Exception as exc:
            logger.debug("Could not fetch funding data for %s: %s", product.symbol, exc)
        return product

    def get_rest_quote(self, symbol: str) -> Quote | None:
        try:
            pid = normalize_symbol(symbol)
            base = cast("CoinbaseRestServiceBase", self)
            client = cast("CoinbaseClient", base.client)
            response = client.get_ticker(pid)
        except Exception as exc:
            logger.error("Failed to get quote for %s: %s", symbol, exc)
            return None
        return to_quote({"symbol": pid, **(response or {})})

    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> list[Any]:
        base = cast("CoinbaseRestServiceBase", self)
        client = cast("CoinbaseClient", base.client)
        data = client.get_candles(normalize_symbol(symbol), granularity, limit) or {}
        items = data.get("candles") or data.get("data") or []
        return [to_candle(candle) for candle in items]

    def get_perpetuals(self) -> list[Product]:
        return self.list_products(market=MarketType.PERPETUAL)


__all__ = ["ProductRestMixin"]
