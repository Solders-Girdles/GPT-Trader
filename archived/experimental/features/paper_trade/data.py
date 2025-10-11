"""
Real-time data fetching for paper trading.

Completely self-contained - no external dependencies.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from bot_v2.data_providers import get_data_provider

logger = logging.getLogger(__name__)


class DataFeed:
    """Real-time data feed for paper trading."""

    def __init__(self, symbols: list[str], lookback_days: int = 30) -> None:
        """
        Initialize data feed.

        Args:
            symbols: List of symbols to track
            lookback_days: Days of historical data to maintain
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.last_update: dict[str, datetime] = {}
        self.update_interval = 60  # seconds

        # Initialize historical data
        self._initialize_historical()

    def _initialize_historical(self) -> None:
        """Load initial historical data for all symbols."""
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        for symbol in self.symbols:
            try:
                provider = get_data_provider()
                data = provider.get_historical_data(
                    symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
                )

                if not data.empty:
                    # Standardize columns
                    data.columns = data.columns.str.lower()
                    self.data_cache[symbol] = data
                    self.last_update[symbol] = datetime.now()
            except Exception as exc:
                logger.warning(
                    "Unable to load historical data for %s: %s", symbol, exc, exc_info=True
                )
                self.data_cache[symbol] = pd.DataFrame()

    def get_latest_price(self, symbol: str) -> float | None:
        """
        Get latest price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price or None if not available
        """
        if symbol not in self.data_cache or self.data_cache[symbol].empty:
            return None

        # During market hours, fetch real-time quote
        if self._is_market_hours():
            try:
                provider = get_data_provider()
                price = provider.get_current_price(symbol)
                if price:
                    return price
                return self.data_cache[symbol]["close"].iloc[-1]
            except Exception as exc:
                logger.debug("Realtime price fetch failed for %s: %s", symbol, exc, exc_info=True)

        # Return last close price
        return float(self.data_cache[symbol]["close"].iloc[-1])

    def get_historical(self, symbol: str, periods: int = None) -> pd.DataFrame:
        """
        Get historical data for a symbol.

        Args:
            symbol: Stock symbol
            periods: Number of periods to return (None for all)

        Returns:
            DataFrame with OHLCV data
        """
        if symbol not in self.data_cache:
            return pd.DataFrame()

        data = self.data_cache[symbol]
        if periods and len(data) > periods:
            return data.iloc[-periods:].copy()
        return data.copy()

    def update(self, symbol: str = None) -> None:
        """
        Update data for symbol(s).

        Args:
            symbol: Specific symbol to update (None for all)
        """
        symbols_to_update = [symbol] if symbol else self.symbols

        for sym in symbols_to_update:
            # Check if update needed
            if sym in self.last_update:
                time_since_update = (datetime.now() - self.last_update[sym]).seconds
                if time_since_update < self.update_interval:
                    continue

            try:
                # Fetch latest data
                provider = get_data_provider()
                end = datetime.now()
                start = end - timedelta(days=1)

                new_data = provider.get_historical_data(
                    sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
                )

                if not new_data.empty:
                    # Aggregate to daily if needed
                    new_data.columns = new_data.columns.str.lower()

                    # Update cache
                    if sym in self.data_cache and not self.data_cache[sym].empty:
                        # Append new data
                        self.data_cache[sym] = pd.concat(
                            [
                                self.data_cache[sym][:-1],  # All but last row
                                new_data.tail(1),  # Latest data
                            ]
                        )

                        # Maintain lookback window
                        if len(self.data_cache[sym]) > self.lookback_days:
                            self.data_cache[sym] = self.data_cache[sym].iloc[-self.lookback_days :]
                    else:
                        self.data_cache[sym] = new_data

                    self.last_update[sym] = datetime.now()

            except Exception as exc:
                logger.warning("Unable to update data for %s: %s", sym, exc, exc_info=True)

    def _is_market_hours(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market hours (simplified check)
        """
        now = datetime.now()
        weekday = now.weekday()

        # Skip weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Simple market hours check (9:30 AM - 4:00 PM ET)
        # This is simplified - real implementation would handle timezones
        hour = now.hour
        minute = now.minute

        market_open = (hour == 9 and minute >= 30) or (hour > 9)
        market_close = hour < 16

        return market_open and market_close

    def add_symbol(self, symbol: str) -> None:
        """Add a new symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._initialize_historical()

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from tracking."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.data_cache:
                del self.data_cache[symbol]
            if symbol in self.last_update:
                del self.last_update[symbol]
