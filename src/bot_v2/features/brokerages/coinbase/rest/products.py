"""Product discovery helpers for Coinbase REST service."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ...core.interfaces import MarketType, Product, Quote
from ..models import normalize_symbol, to_candle, to_product, to_quote
from .base import logger


class ProductRestMixin:
    """Product discovery and market data helpers."""

    def list_products(self, market: MarketType | None = None) -> list[Product]:
        try:
            response = self.client.get_products() or {}
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
            product = self.product_catalog.get(self.client, normalized)
        except Exception as exc:
            logger.debug("Product catalog lookup failed for %s: %s", normalized, exc, exc_info=True)
            product = None

        if product is None:
            try:
                if hasattr(self.client, "get_product"):
                    data = self.client.get_product(normalized) or {}
                else:  # pragma: no cover - legacy endpoint
                    data = self.client.get(self.endpoints.get_product(normalized)) or {}
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
            funding_rate, next_funding_time = self.product_catalog.get_funding(
                self.client, product.symbol
            )
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
            response = self.client.get_ticker(pid)
        except Exception as exc:
            logger.error("Failed to get quote for %s: %s", symbol, exc)
            return None
        return to_quote({"symbol": pid, **(response or {})})

    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> list[Any]:
        data = self.client.get_candles(normalize_symbol(symbol), granularity, limit) or {}
        items = data.get("candles") or data.get("data") or []
        return [to_candle(candle) for candle in items]

    def get_perpetuals(self) -> list[Product]:
        return self.list_products(market=MarketType.PERPETUAL)


__all__ = ["ProductRestMixin"]
