"""Pricing helpers for the Coinbase data provider."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_provider")


class CoinbasePricingMixin:
    """Spot price and multi-symbol helpers."""

    enable_streaming: bool
    cache_ttl: int
    ticker_cache: Any

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price from Coinbase, using streaming cache when available.
        """
        normalized_symbol = self._normalize_symbol(symbol)

        try:
            if self.enable_streaming and self.ticker_cache:
                ticker = self.ticker_cache.get(normalized_symbol)
                if ticker and not self.ticker_cache.is_stale(normalized_symbol, self.cache_ttl):
                    last_price = ticker.last
                    if last_price is not None:
                        logger.debug(
                            "Using WebSocket price",
                            symbol=normalized_symbol,
                            price=float(last_price),
                            source="websocket",
                        )
                        return float(last_price)

            quote = self.adapter.get_quote(normalized_symbol)
            price_source = quote.last or getattr(quote, "ask", None) or getattr(quote, "bid", None)
            if price_source is None:
                raise ValueError("Quote did not contain a price")
            price = float(price_source)
            logger.debug(
                "Using REST API price",
                symbol=normalized_symbol,
                price=price,
                source="rest",
            )
            return price

        except Exception as exc:
            logger.error(
                "Error fetching current price",
                symbol=normalized_symbol,
                operation="price_fetch",
                status="error",
                error=str(exc),
            )
            return 100.0

    def get_multiple_symbols(
        self, symbols: list[str], period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols with optional streaming subscription.
        """
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]
        self._subscribe_streaming(normalized_symbols)
        return {symbol: self.get_historical_data(symbol, period) for symbol in symbols}

    @staticmethod
    def is_market_open() -> bool:
        """Crypto markets trade 24/7."""
        return True


__all__ = ["CoinbasePricingMixin"]
