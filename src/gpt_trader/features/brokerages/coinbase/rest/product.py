"""
Product and market data mixin for Coinbase REST service.
"""

from __future__ import annotations

from typing import List, Optional

from gpt_trader.features.brokerages.coinbase.models import to_candle, to_product, to_quote
from gpt_trader.features.brokerages.core.interfaces import Candle, Product, Quote
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_product")


class ProductRestMixin:
    """Mixin for product and market data operations."""

    def list_products(self) -> List[Product]:
        """List all available products."""
        try:
            response = self.client.get_products()
            products = []
            # Handle both list and dict response shapes
            items = response if isinstance(response, list) else response.get("products", [])
            for p in items:
                try:
                    products.append(to_product(p))
                except Exception:
                    continue
            return products
        except Exception:
            return []

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get details of a single product."""
        try:
            # Try catalog first for enrichment
            if hasattr(self, "product_catalog") and self.product_catalog:
                try:
                    # Assuming catalog has a get method that takes client and symbol
                    # based on test_funding_enrichment_uses_product_catalog
                    product = self.product_catalog.get(self.client, product_id)

                    # Enrich with funding if available
                    if hasattr(self.product_catalog, "get_funding"):
                        try:
                            funding_rate, next_funding = self.product_catalog.get_funding(
                                self.client, product_id
                            )
                            product.funding_rate = funding_rate
                            product.next_funding_time = next_funding
                        except Exception:
                            pass
                    return product
                except Exception as e:
                    logger.debug("ProductRestMixin catalog get failed: %s", e)
                    pass

            p = self.client.get_product(product_id)
            return to_product(p)
        except Exception:
            return None

    def get_rest_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote (bid/ask/last) for a symbol via REST."""
        try:
            # This might need a specific endpoint or ticker
            ticker = self.client.get_product_ticker(symbol)
            return to_quote(ticker)
        except Exception:
            return None

    def get_candles(self, symbol: str, **kwargs) -> List[Candle]:
        """Get historical OHLCV candles for a symbol."""
        try:
            response = self.client.get_candles(symbol, **kwargs)
            candles = []
            items = response.get("candles", [])
            for c in items:
                try:
                    candles.append(to_candle(c))
                except Exception:
                    continue
            return candles
        except Exception:
            return []

    def get_perpetuals(self) -> List[Product]:
        """List perpetual products."""
        products = self.list_products()
        return [p for p in products if p.market_type == "PERPETUAL"]

    def get_futures(self) -> List[Product]:
        """List futures products."""
        products = self.list_products()
        return [p for p in products if p.market_type == "FUTURE"]
