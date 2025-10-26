"""Historical data helpers for the Coinbase data provider."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_provider")


class CoinbaseHistoricalDataMixin:
    """Historical data retrieval with REST fallback and mock generation."""

    _historical_cache: dict[str, tuple[pd.DataFrame, datetime]]
    _cache_duration: timedelta

    def _initialize_historical_cache(self, cache_ttl: int) -> None:
        self._historical_cache = {}
        self._cache_duration = timedelta(seconds=cache_ttl)

    def get_historical_data(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical candles from Coinbase with caching and mock fallback.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        cache_key = f"{normalized_symbol}_{period}_{interval}"

        if cache_key in self._historical_cache:
            cached_data, cached_time = self._historical_cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(
                    "Using cached historical data",
                    symbol=normalized_symbol,
                    period=period,
                    interval=interval,
                )
                return cached_data

        granularity_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR",
            "2h": "TWO_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY",
        }

        granularity = granularity_map.get(interval, "ONE_DAY")

        try:
            days = int(period.rstrip("d")) if "d" in period else 60
        except ValueError:
            days = 60

        if interval == "1d":
            limit = days
        elif interval == "1h":
            limit = min(days * 24, 300)
        elif interval == "5m":
            limit = min(days * 24 * 12, 300)
        else:
            limit = 200

        try:
            logger.info(
                "Fetching historical candles",
                symbol=normalized_symbol,
                granularity=granularity,
                limit=limit,
                operation="historical_fetch",
            )
            candles = self.adapter.get_candles(normalized_symbol, granularity, limit)

            if not candles:
                logger.warning(
                    "No Coinbase candles returned; falling back to mock data",
                    symbol=normalized_symbol,
                    operation="historical_fetch",
                    status="empty",
                )
                return self._get_mock_data(symbol, period)

            rows: list[dict[str, Any]] = [
                {
                    "timestamp": candle.ts,
                    "Open": float(candle.open),
                    "High": float(candle.high),
                    "Low": float(candle.low),
                    "Close": float(candle.close),
                    "Volume": float(candle.volume),
                }
                for candle in candles
            ]

            df = pd.DataFrame(rows)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            self._historical_cache[cache_key] = (df, datetime.now())
            logger.info(
                "Retrieved historical data",
                symbol=normalized_symbol,
                rows=len(df),
                operation="historical_fetch",
                status="success",
            )
            return df

        except Exception as exc:
            logger.error(
                "Error fetching Coinbase historical data",
                symbol=normalized_symbol,
                operation="historical_fetch",
                status="error",
                error=str(exc),
            )
            logger.info(
                "Falling back to mock historical data",
                symbol=normalized_symbol,
                operation="historical_fetch",
                status="fallback",
            )
            return self._get_mock_data(symbol, period)

    def _get_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate deterministic mock data for tests and fallbacks."""
        try:
            days = int(period.rstrip("d")) if "d" in period else 60
        except ValueError:
            days = 60
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq="D")

        np.random.seed(hash(symbol) % 2**32)

        base_price = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0
        returns = np.random.normal(0.002, 0.03, days)
        prices = base_price * np.exp(np.cumsum(returns))

        opens = prices * (1 + np.random.uniform(-0.02, 0.02, days))
        high_factor = 1 + np.abs(np.random.uniform(0, 0.04, days))
        highs = np.maximum(opens, prices) * high_factor
        low_factor = 1 - np.abs(np.random.uniform(0, 0.04, days))
        lows = np.minimum(opens, prices) * low_factor

        data = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": np.random.randint(10_000_000, 100_000_000, days),
            },
            index=dates,
        )
        return data


__all__ = ["CoinbaseHistoricalDataMixin"]
