"""
Data provider abstraction retained for legacy equities support.

Production market data flows through the Coinbase brokerage/perps stack.  This
module now offers only the yfinance-backed provider (with mock fallbacks) so
older tutorials and tests can keep running without third-party dependencies.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for all data providers"""

    @abstractmethod
    def get_historical_data(
        self, symbol: str, period: str = "60d", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a symbol

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            period: Time period (e.g., "60d", "1y", "5y")
            interval: Data interval (e.g., "1d", "1h", "5m")

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        pass

    @abstractmethod
    def get_multiple_symbols(
        self, symbols: list[str], period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider - default implementation"""

    def __init__(self) -> None:
        self._yfinance = None
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(minutes=5)

    @property
    def yf(self):
        """Lazy load yfinance"""
        if self._yfinance is None:
            try:
                import yfinance as yf

                self._yfinance = yf
            except ImportError:
                raise ImportError("yfinance not installed. Run: pip install yfinance")
        return self._yfinance

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]

    def get_historical_data(
        self, symbol: str, period: str = "60d", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        cache_key = f"{symbol}_{period}_{interval}"

        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            ticker = self.yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return self._get_mock_data(symbol, period)

            # Cache the data
            self._cache[cache_key] = data
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_mock_data(symbol, period)

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance"""
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info
            return info.get("currentPrice", info.get("regularMarketPrice", 100.0))
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 100.0

    def get_multiple_symbols(
        self, symbols: list[str], period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, period)
        return result

    def is_market_open(self) -> bool:
        """Check if US market is open"""
        now = datetime.now()
        # Simple check - US market hours (9:30 AM - 4:00 PM ET)
        # This is simplified - doesn't account for holidays
        if now.weekday() >= 5:  # Weekend
            return False
        hour = now.hour
        return 9 <= hour < 16

    def _get_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate mock data as fallback"""
        days = int(period.rstrip("d")) if "d" in period else 60
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Generate realistic-looking price data
        import numpy as np

        np.random.seed(hash(symbol) % 2**32)

        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * np.random.uniform(0.98, 1.02, days),
                "High": prices * np.random.uniform(1.01, 1.05, days),
                "Low": prices * np.random.uniform(0.95, 0.99, days),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, days),
            },
            index=dates,
        )

        return data


class MockProvider(DataProvider):
    """Mock data provider for testing"""

    def __init__(self, data_dir: str = None) -> None:
        self.data_dir = data_dir or "tests/fixtures/market_data"
        self._mock_data = {}
        self._load_mock_data()

    def _load_mock_data(self) -> None:
        """Load mock data from fixtures"""
        # Try to load from JSON fixtures
        fixture_file = os.path.join(self.data_dir, "mock_data.json")
        if os.path.exists(fixture_file):
            with open(fixture_file) as f:
                self._mock_data = json.load(f)

    def get_historical_data(
        self, symbol: str, period: str = "60d", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get mock historical data"""
        # Generate deterministic mock data
        days = int(period.rstrip("d")) if "d" in period else 60
        # Use fixed end date for consistency in tests
        end_date = datetime(2024, 3, 1)
        dates = pd.date_range(end=end_date, periods=days, freq="D")

        # Use symbol hash for consistent data
        import numpy as np

        np.random.seed(hash(symbol) % 2**32)

        base_price = 100.0
        returns = np.random.normal(0.001, 0.015, days)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC data with proper relationships
        # Open is slightly off from previous close
        opens = prices * (1 + np.random.uniform(-0.01, 0.01, days))

        # High should be max of open, close, and a bit higher
        high_factor = 1 + np.abs(np.random.uniform(0, 0.02, days))
        highs = np.maximum(opens, prices) * high_factor

        # Low should be min of open, close, and a bit lower
        low_factor = 1 - np.abs(np.random.uniform(0, 0.02, days))
        lows = np.minimum(opens, prices) * low_factor

        data = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, days),
            },
            index=dates,
        )

        return data

    def get_current_price(self, symbol: str) -> float:
        """Get mock current price"""
        # Return consistent price based on symbol
        prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0, "TSLA": 250.0}
        return prices.get(symbol, 100.0)

    def get_multiple_symbols(
        self, symbols: list[str], period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        """Get mock data for multiple symbols"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, period)
        return result

    def is_market_open(self) -> bool:
        """Mock market hours"""
        return True  # Always open for testing


# Global provider instance
_provider_instance = None


def get_data_provider(provider_type: str = None) -> DataProvider:
    """
    Factory function to get appropriate data provider

    Args:
        provider_type: Optional provider type ('yfinance', 'mock', 'coinbase')
                      If None, auto-detects based on environment

    Returns:
        DataProvider instance
    """
    global _provider_instance

    # If TESTING is explicitly enabled, always use a fresh MockProvider
    testing_mode = os.environ.get("TESTING", "").lower() == "true"
    if provider_type is None and testing_mode:
        _provider_instance = MockProvider()
        logger.info("Using mock data provider (TESTING=true)")
        return _provider_instance

    if _provider_instance is not None and provider_type is None:
        return _provider_instance

    # Auto-detect provider type if not specified
    if provider_type is None:
        if os.environ.get("TESTING", "").lower() == "true":
            provider_type = "mock"
        elif (
            os.environ.get("COINBASE_USE_REAL_DATA", "").lower() == "true"
            or os.environ.get("COINBASE_USE_REAL_DATA") == "1"
        ):
            provider_type = "coinbase"
        else:
            provider_type = "yfinance"

    # Handle legacy aliases
    if provider_type == "alpaca":
        logger.warning("Alpaca data provider has been deprecated; falling back to yfinance")
        provider_type = "yfinance"

    # Create appropriate provider
    if provider_type == "mock":
        _provider_instance = MockProvider()
    elif provider_type == "coinbase":
        # Lazy import to avoid circular dependencies
        from .coinbase_provider import CoinbaseDataProvider

        _provider_instance = CoinbaseDataProvider()
    else:
        _provider_instance = YFinanceProvider()

    logger.info(f"Using {provider_type} data provider")
    return _provider_instance


def set_data_provider(provider: DataProvider) -> None:
    """Set custom data provider instance"""
    global _provider_instance
    _provider_instance = provider


# Export main interface
__all__ = [
    "DataProvider",
    "YFinanceProvider",
    "MockProvider",
    "get_data_provider",
    "set_data_provider",
]
