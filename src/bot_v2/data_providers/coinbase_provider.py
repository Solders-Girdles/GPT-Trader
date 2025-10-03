"""
Coinbase Data Provider - Real market data from Coinbase API.

This provider integrates with the existing Coinbase brokerage adapter
to fetch real market data instead of using mock data. It supports both
REST API calls and WebSocket streaming for real-time updates.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from bot_v2.data_providers import DataProvider
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    TickerCache,
)
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket

logger = logging.getLogger(__name__)


class CoinbaseDataProvider(DataProvider):
    """
    Coinbase real-time data provider using the Coinbase API.

    This provider can operate in two modes:
    1. REST-only mode: Fetches data via REST API calls (default)
    2. Streaming mode: Uses WebSocket for real-time updates with REST fallback

    Configuration via environment variables:
    - COINBASE_USE_REAL_DATA: Set to "1" to use real API data
    - COINBASE_ENABLE_STREAMING: Set to "1" to enable WebSocket streaming
    - COINBASE_DATA_CACHE_TTL: Cache TTL in seconds (default: 5)
    """

    def __init__(
        self,
        client: CoinbaseClient | None = None,
        adapter: CoinbaseBrokerage | None = None,
        enable_streaming: bool = False,
        cache_ttl: int = 5,
    ) -> None:
        """
        Initialize Coinbase data provider.

        Args:
            client: Optional CoinbaseClient instance (creates one if not provided)
            adapter: Optional CoinbaseBrokerage instance (creates one if not provided)
            enable_streaming: Enable WebSocket streaming for real-time data
            cache_ttl: Cache time-to-live in seconds
        """
        # Initialize client and adapter if not provided
        if client is None:
            # Determine if we're using sandbox or production (force production for live data)
            sandbox = os.environ.get("COINBASE_SANDBOX", "0") == "1"
            if sandbox:
                logger.warning(
                    "COINBASE_SANDBOX=1 detected while requesting real market data. "
                    "Defaulting to production endpoints."
                )
                sandbox = False

            api_mode = (os.environ.get("COINBASE_API_MODE", "advanced") or "advanced").lower()
            if api_mode == "exchange":
                logger.warning(
                    "COINBASE_API_MODE=exchange detected while requesting real market data. "
                    "Defaulting to Advanced Trade production endpoints."
                )
                api_mode = "advanced"

            # Set base URL based on mode
            if api_mode == "advanced":
                base_url = "https://api.coinbase.com"
            else:
                base_url = "https://api.exchange.coinbase.com"

            # Create client without auth for public market data
            # Auth would be needed for private endpoints, but market data is public
            self.client = CoinbaseClient(base_url=base_url, auth=None, api_mode=api_mode)
        else:
            self.client = client

        if adapter is None:
            # Create adapter with minimal config for market data
            config = APIConfig(
                api_key="",  # Not needed for public data
                api_secret="",
                passphrase="",
                base_url=self.client.base_url,
                ws_url="wss://advanced-trade-ws.coinbase.com",
                api_mode=self.client.api_mode,
                sandbox=sandbox,
                auth_type="HMAC",  # Default auth type for public data
                enable_derivatives=False,  # Not needed for market data
            )
            self.adapter = CoinbaseBrokerage(config=config)
        else:
            self.adapter = adapter

        # Configure streaming
        self.enable_streaming = (
            enable_streaming or os.environ.get("COINBASE_ENABLE_STREAMING", "0") == "1"
        )
        self.cache_ttl = cache_ttl

        # Initialize ticker service for streaming if enabled
        self.ticker_service: CoinbaseTickerService | None = None
        self.ticker_cache: TickerCache | None = None

        if self.enable_streaming:
            self._setup_streaming()

        # Cache for historical data
        self._historical_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._cache_duration = timedelta(seconds=cache_ttl)

        streaming_status = "enabled" if self.enable_streaming else "disabled"
        logger.info(f"CoinbaseDataProvider initialized (streaming={streaming_status})")

    def _setup_streaming(self) -> None:
        """Setup WebSocket streaming for real-time data."""
        try:
            self.ticker_cache = TickerCache()

            def ws_factory():
                ws = CoinbaseWebSocket()
                ws.connect()
                return ws

            # Start with empty symbols, will subscribe when needed
            self.ticker_service = CoinbaseTickerService(
                websocket_factory=ws_factory,
                symbols=[],
                cache=self.ticker_cache,
                on_update=self._on_ticker_update,
            )

            logger.info("WebSocket streaming setup complete")
        except Exception as e:
            logger.warning(f"Failed to setup streaming, falling back to REST: {e}")
            self.enable_streaming = False

    def _on_ticker_update(self, symbol: str, ticker: Any) -> None:
        """Callback for ticker updates from WebSocket."""
        logger.debug(
            f"Ticker update for {symbol}: bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}"
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for Coinbase.

        Converts equity-style symbols (AAPL, BTC) to Coinbase format (BTC-USD by default,
        BTC-PERP when derivatives are enabled).
        """
        # If already in Coinbase format, return as-is
        if "-" in symbol:
            return symbol.upper()

        # Default the mapping to spot pairs; append -PERP only when derivatives are enabled.
        symbol_upper = symbol.upper()
        derivatives_enabled = os.environ.get("COINBASE_ENABLE_DERIVATIVES", "0") == "1"

        spot_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "DOGE": "DOGE-USD",
            "AVAX": "AVAX-USD",
            "LINK": "LINK-USD",
            "ADA": "ADA-USD",
            "DOT": "DOT-USD",
            "MATIC": "MATIC-USD",
            "UNI": "UNI-USD",
        }

        perp_map = {
            "BTC": "BTC-PERP",
            "ETH": "ETH-PERP",
            "SOL": "SOL-PERP",
            "DOGE": "DOGE-PERP",
            "AVAX": "AVAX-PERP",
            "LINK": "LINK-PERP",
            "ADA": "ADA-PERP",
            "DOT": "DOT-PERP",
            "MATIC": "MATIC-PERP",
            "UNI": "UNI-PERP",
        }

        # Allow explicit override of the quote currency (e.g., USDC) for spot pairs.
        default_quote = os.environ.get("COINBASE_DEFAULT_QUOTE", "USD").upper()
        if default_quote not in {"USD", "USDC", "USDT"}:
            default_quote = "USD"

        if derivatives_enabled and symbol_upper in perp_map:
            return perp_map[symbol_upper]

        if symbol_upper in spot_map:
            base, _ = spot_map[symbol_upper].split("-")
            return f"{base}-{default_quote}"

        # Default to quote pair for spot trading
        return f"{symbol_upper}-{default_quote}"

    def _build_cache_key(
        self,
        symbol: str,
        period: str,
        interval: str,
        start: datetime | str | None,
        end: datetime | str | None,
    ) -> str:
        """Build cache key that distinguishes explicit date ranges."""
        start_key = self._format_cache_bound(start)
        end_key = self._format_cache_bound(end)
        if not start_key and not end_key:
            return f"{symbol}_{period}_{interval}"
        return f"{symbol}_{period}_{interval}_{start_key}_{end_key}"

    @staticmethod
    def _format_cache_bound(bound: datetime | str | None) -> str:
        if bound is None:
            return ""
        if isinstance(bound, datetime):
            return bound.isoformat()
        return str(bound)

    def _resolve_period_days(
        self,
        period: str,
        start: datetime | str | None,
        end: datetime | str | None,
    ) -> int:
        """Determine the requested window length in days."""
        start_dt = self._coerce_datetime(start)
        end_dt = self._coerce_datetime(end)
        if start_dt is not None and end_dt is not None:
            delta_days = max(int((end_dt - start_dt).days) + 1, 1)
            return delta_days

        return self._parse_period_days(period)

    @staticmethod
    def _coerce_datetime(value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return pd.to_datetime(value).to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _parse_period_days(period: str) -> int:
        try:
            if period.endswith("d"):
                return max(int(period[:-1]), 1)
        except (AttributeError, ValueError):
            return 60
        return 60

    def _fallback_period(
        self,
        period: str,
        start: datetime | str | None,
        end: datetime | str | None,
        period_days: int | None,
    ) -> str:
        """Return a period string suitable for mock data fallbacks."""
        if period:
            return period
        if period_days is None:
            period_days = self._resolve_period_days(period, start, end)
        return f"{period_days}d"

    def get_historical_data(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "1d",
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Get historical price data from Coinbase.

        Args:
            symbol: Symbol (e.g., "BTC", "ETH", "BTC-PERP")
            period: Time period (e.g., "60d", "30d", "7d")
            interval: Data interval (maps to Coinbase granularities)
                      "1d" -> "ONE_DAY", "1h" -> "ONE_HOUR", "5m" -> "FIVE_MINUTE"

        Returns:
            DataFrame with OHLCV data
        """
        normalized_symbol = self._normalize_symbol(symbol)
        cache_key = self._build_cache_key(normalized_symbol, period, interval, start, end)

        # Check cache
        if cache_key in self._historical_cache:
            cached_data, cached_time = self._historical_cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Using cached data for {normalized_symbol}")
                return cached_data

        try:
            # Map interval to Coinbase granularity
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

            # Calculate limit based on requested window
            period_days = self._resolve_period_days(period, start, end)
            if interval == "1d":
                limit = period_days
            elif interval == "1h":
                limit = min(period_days * 24, 300)  # Coinbase has limits
            elif interval == "5m":
                limit = min(period_days * 24 * 12, 300)
            else:
                limit = 200  # Default

            # Fetch candles from Coinbase
            logger.info(f"Fetching {limit} candles for {normalized_symbol} at {granularity}")
            candles = self.adapter.get_candles(normalized_symbol, granularity, limit)

            if not candles:
                logger.warning(f"No data returned for {normalized_symbol}, using mock data")
                fallback_period = self._fallback_period(period, start, end, period_days)
                return self._get_mock_data(symbol, fallback_period)

            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append(
                    {
                        "timestamp": candle.ts,
                        "Open": float(candle.open),
                        "High": float(candle.high),
                        "Low": float(candle.low),
                        "Close": float(candle.close),
                        "Volume": float(candle.volume),
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Cache the result
            self._historical_cache[cache_key] = (df, datetime.now())

            logger.info(f"Retrieved {len(df)} bars of historical data for {normalized_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching Coinbase data for {normalized_symbol}: {e}")
            logger.info("Falling back to mock data")
            fallback_period = self._fallback_period(period, start, end, period_days=None)
            return self._get_mock_data(symbol, fallback_period)

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price from Coinbase.

        Uses WebSocket cache if available and fresh, otherwise REST API.
        """
        normalized_symbol = self._normalize_symbol(symbol)

        try:
            # Try WebSocket cache first if streaming is enabled
            if self.enable_streaming and self.ticker_cache:
                ticker = self.ticker_cache.get(normalized_symbol)
                if ticker and not self.ticker_cache.is_stale(normalized_symbol, self.cache_ttl):
                    logger.debug(f"Using WebSocket price for {normalized_symbol}: {ticker.last}")
                    return float(ticker.last)

            # Fall back to REST API
            quote = self.adapter.get_quote(normalized_symbol)
            price = float(quote.last)
            logger.debug(f"Using REST API price for {normalized_symbol}: {price}")
            return price

        except Exception as e:
            logger.error(f"Error fetching current price for {normalized_symbol}: {e}")
            # Return a reasonable default
            return 100.0

    def get_multiple_symbols(
        self, symbols: list[str], period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        If streaming is enabled, subscribes to all symbols for real-time updates.
        """
        # Subscribe to WebSocket updates if streaming is enabled
        if self.enable_streaming and self.ticker_service:
            normalized_symbols = [self._normalize_symbol(s) for s in symbols]
            try:
                # Update the ticker service with new symbols
                self.ticker_service._symbols = normalized_symbols
                if not self.ticker_service._thread or not self.ticker_service._thread.is_alive():
                    self.ticker_service.start()
                logger.info(
                    f"Subscribed to WebSocket updates for {len(normalized_symbols)} symbols"
                )
            except Exception as e:
                logger.warning(f"Failed to subscribe to WebSocket: {e}")

        # Fetch historical data for each symbol
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, period)

        return result

    def is_market_open(self) -> bool:
        """
        Check if crypto market is open.

        Crypto markets are 24/7, so always returns True.
        For traditional hours, could check specific exchange hours.
        """
        # Crypto markets are always open
        return True

    def _get_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Generate mock data as fallback.

        This maintains the same deterministic behavior as MockProvider
        for testing consistency.
        """
        days = int(period.rstrip("d")) if "d" in period else 60
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq="D")

        # Use symbol hash for consistent data
        np.random.seed(hash(symbol) % 2**32)

        # Generate more realistic crypto-like volatility
        base_price = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0
        returns = np.random.normal(0.002, 0.03, days)  # Higher volatility for crypto
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC data
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
                "Volume": np.random.randint(10000000, 100000000, days),
            },
            index=dates,
        )

        return data

    def start_streaming(self) -> None:
        """Start WebSocket streaming if configured."""
        if self.enable_streaming and self.ticker_service:
            self.ticker_service.start()
            logger.info("WebSocket streaming started")

    def stop_streaming(self) -> None:
        """Stop WebSocket streaming."""
        if self.ticker_service:
            self.ticker_service.stop()
            logger.info("WebSocket streaming stopped")

    def __enter__(self):
        """Context manager entry - starts streaming if enabled."""
        self.start_streaming()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops streaming."""
        self.stop_streaming()


def create_coinbase_provider(
    use_real_data: bool | None = None, enable_streaming: bool | None = None
) -> DataProvider:
    """
    Factory function to create appropriate Coinbase data provider.

    Args:
        use_real_data: Use real API data (None = check env var)
        enable_streaming: Enable WebSocket streaming (None = check env var)

    Returns:
        CoinbaseDataProvider if real data is enabled, MockProvider otherwise
    """
    # Check environment configuration
    if use_real_data is None:
        use_real_data = os.environ.get("COINBASE_USE_REAL_DATA", "0") == "1"

    if not use_real_data:
        # Return mock provider for testing
        from bot_v2.data_providers import MockProvider

        logger.info("Using MockProvider (COINBASE_USE_REAL_DATA=0)")
        return MockProvider()

    # Create real Coinbase provider
    if enable_streaming is None:
        enable_streaming = os.environ.get("COINBASE_ENABLE_STREAMING", "0") == "1"

    logger.info(f"Creating CoinbaseDataProvider (streaming={enable_streaming})")
    return CoinbaseDataProvider(enable_streaming=enable_streaming)
