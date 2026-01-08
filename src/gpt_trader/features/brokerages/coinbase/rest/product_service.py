"""Product and market data service for Coinbase REST API.

This service handles product and market data operations with explicit
dependencies injected via constructor, replacing the ProductRestMixin.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from gpt_trader.core import Candle, Product, Quote
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import to_candle, to_product, to_quote
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_product")


class ProductService:
    """Handles product and market data operations.

    Dependencies:
        client: CoinbaseClient for API calls
        product_catalog: ProductCatalog for caching and enrichment
        market_data: MarketDataService for real-time prices
    """

    def __init__(
        self,
        *,
        client: CoinbaseClient,
        product_catalog: ProductCatalog,
        market_data: MarketDataService,
    ) -> None:
        self._client = client
        self._product_catalog = product_catalog
        self._market_data = market_data

    def list_products(self) -> list[Product]:
        """List all available products."""
        try:
            response = self._client.get_products()
            products = []
            # Handle both list and dict response shapes
            items = response if isinstance(response, list) else response.get("products", [])
            for p in items:
                try:
                    products.append(to_product(p))
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug("Skipping product due to parse error: %s", e)
                    continue
            return products
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error listing products: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error listing products: %s", e, exc_info=True)
            return []

    def get_product(self, product_id: str) -> Product | None:
        """Get details of a single product."""
        try:
            # Try catalog first for enrichment
            if self._product_catalog:
                try:
                    # Assuming catalog has a get method that takes client and symbol
                    # based on test_funding_enrichment_uses_product_catalog
                    product = self._product_catalog.get(self._client, product_id)

                    # Enrich with funding if available
                    if hasattr(self._product_catalog, "get_funding"):
                        try:
                            funding_rate, next_funding = self._product_catalog.get_funding(
                                self._client, product_id
                            )
                            product.funding_rate = funding_rate
                            product.next_funding_time = next_funding
                        except (KeyError, ValueError, AttributeError) as e:
                            logger.debug("Funding enrichment failed for %s: %s", product_id, e)
                    return cast(Product, product)
                except (KeyError, ValueError) as e:
                    logger.debug("ProductService catalog get failed: %s", e)

            p = self._client.get_product(product_id)
            return to_product(p)
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error getting product %s: %s", product_id, e)
            return None
        except (KeyError, ValueError) as e:
            logger.debug("Product %s not found or invalid: %s", product_id, e)
            return None
        except Exception as e:
            logger.error("Unexpected error getting product %s: %s", product_id, e, exc_info=True)
            return None

    def get_rest_quote(self, symbol: str) -> Quote | None:
        """Get current quote (bid/ask/last) for a symbol via REST."""
        try:
            # This might need a specific endpoint or ticker
            ticker = self._client.get_product_ticker(symbol)
            return to_quote(ticker)
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error getting quote for %s: %s", symbol, e)
            return None
        except (KeyError, ValueError) as e:
            logger.debug("Quote parse error for %s: %s", symbol, e)
            return None
        except Exception as e:
            logger.error("Unexpected error getting quote for %s: %s", symbol, e, exc_info=True)
            return None

    def get_candles(self, symbol: str, **kwargs: Any) -> list[Candle]:
        """Get historical OHLCV candles for a symbol."""
        try:
            response = self._client.get_candles(symbol, **kwargs)
            candles = []
            items = response.get("candles", [])
            for c in items:
                try:
                    candles.append(to_candle(c))
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug("Skipping candle due to parse error: %s", e)
                    continue
            return candles
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error getting candles for %s: %s", symbol, e)
            return []
        except Exception as e:
            logger.error("Unexpected error getting candles for %s: %s", symbol, e, exc_info=True)
            return []

    def get_perpetuals(self) -> list[Product]:
        """List perpetual products (INTX)."""
        products = self.list_products()
        return [p for p in products if p.market_type == "PERPETUAL"]

    def get_futures(self) -> list[Product]:
        """List futures products (includes CFM)."""
        products = self.list_products()
        return [p for p in products if p.market_type == "FUTURE"]

    def get_cfm_products(self) -> list[Product]:
        """List CFM (Coinbase Financial Markets) futures products.

        CFM futures are US-regulated futures with expiration dates.
        They use FUTURE market type and have expiry dates set.

        Returns:
            List of CFM futures products.
        """
        products = self.list_products()
        return [p for p in products if p.market_type == "FUTURE" and p.expiry is not None]

    def get_spot_products(self) -> list[Product]:
        """List spot products.

        Returns:
            List of spot trading products.
        """
        products = self.list_products()
        return [p for p in products if p.market_type == "SPOT"]

    def get_tradeable_products(self, modes: list[str]) -> list[Product]:
        """Get products filtered by trading modes.

        Supports hybrid mode by combining products from multiple markets.

        Args:
            modes: List of trading modes - "spot", "cfm", or both for hybrid.

        Returns:
            List of products matching the specified trading modes.
        """
        products: list[Product] = []

        if "spot" in modes:
            products.extend(self.get_spot_products())

        if "cfm" in modes:
            products.extend(self.get_cfm_products())

        # Remove duplicates while preserving order
        seen = set()
        unique_products = []
        for p in products:
            if p.symbol not in seen:
                seen.add(p.symbol)
                unique_products.append(p)

        return unique_products

    # =========================================================================
    # BrokerProtocol / ExtendedBrokerProtocol methods
    # =========================================================================

    def get_quote(self, symbol: str) -> Quote | None:
        """Get current quote for a symbol (BrokerProtocol).

        Delegates to get_rest_quote() for REST-based quote retrieval.
        """
        return self.get_rest_quote(symbol)

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        """Get ticker data for a product (BrokerProtocol).

        Returns raw ticker dict from the Coinbase API.
        """
        try:
            result: dict[str, Any] = self._client.get_ticker(product_id)
            return result
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error getting ticker for %s: %s", product_id, e)
            return {}
        except Exception as e:
            logger.error("Unexpected error getting ticker for %s: %s", product_id, e, exc_info=True)
            return {}

    def get_mark_price(self, symbol: str) -> Decimal | None:
        """Get current mark price for a symbol (ExtendedBrokerProtocol).

        Uses market data service for real-time mark price.
        """
        try:
            mark = self._market_data.get_mark(symbol)
            return Decimal(str(mark)) if mark is not None else None
        except (ValueError, TypeError) as e:
            logger.debug("Mark price conversion error for %s: %s", symbol, e)
            return None
        except Exception as e:
            logger.error("Unexpected error getting mark price for %s: %s", symbol, e, exc_info=True)
            return None

    def get_tickers(self, product_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get ticker data for multiple products in a single batch request.

        In Advanced API mode, uses the batch best_bid_ask endpoint for efficiency.
        In Exchange API mode, falls back to individual get_ticker calls.

        Args:
            product_ids: List of product symbols (e.g., ["BTC-USD", "ETH-USD"])

        Returns:
            Dict mapping product_id to ticker dict with at least a "price" key.
            Missing products are omitted from the result.
        """
        if not product_ids:
            return {}

        result: dict[str, dict[str, Any]] = {}

        # Try batch API if available (Advanced mode only)
        if self._client.api_mode == "advanced":
            try:
                response = self._client.get_best_bid_ask(product_ids)
                pricebooks = response.get("pricebooks", [])
                for book in pricebooks:
                    product_id = book.get("product_id")
                    if not product_id:
                        continue

                    # Extract best bid and ask
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])

                    best_bid = Decimal(bids[0].get("price", "0")) if bids else Decimal("0")
                    best_ask = Decimal(asks[0].get("price", "0")) if asks else Decimal("0")

                    # Calculate mid-price (used as "price" for strategy)
                    if best_bid > 0 and best_ask > 0:
                        mid_price = (best_bid + best_ask) / 2
                    elif best_ask > 0:
                        mid_price = best_ask
                    elif best_bid > 0:
                        mid_price = best_bid
                    else:
                        continue  # No valid price data

                    result[product_id] = {
                        "price": str(mid_price),
                        "bid": str(best_bid),
                        "ask": str(best_ask),
                        "product_id": product_id,
                    }

                # Log success
                if result:
                    logger.debug(
                        "Batch ticker fetch: %d/%d products via best_bid_ask",
                        len(result),
                        len(product_ids),
                    )
                return result

            except Exception as e:
                # Log and fall through to individual fetches
                logger.warning("Batch ticker fetch failed, falling back to individual: %s", e)

        # Fall back to individual ticker calls (Exchange mode or batch failure)
        for product_id in product_ids:
            try:
                ticker = self._client.get_ticker(product_id)
                if ticker and ticker.get("price"):
                    result[product_id] = ticker
            except Exception as e:
                logger.debug("Failed to get ticker for %s: %s", product_id, e)

        logger.debug(
            "Individual ticker fetch: %d/%d products",
            len(result),
            len(product_ids),
        )
        return result
