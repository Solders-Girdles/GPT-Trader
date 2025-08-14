from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ..exec.base import Broker
from ..logging import get_logger

logger = get_logger("live_data")


@dataclass
class MarketData:
    """Market data for a symbol."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    vwap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "vwap": self.vwap,
        }


class LiveDataManager:
    """Manages real-time market data for live trading."""

    def __init__(self, broker: Broker, symbols: list[str]) -> None:
        """Initialize the data manager."""
        self.broker = broker
        self.symbols = set(symbols)
        self.data_cache: dict[str, MarketData] = {}
        self.historical_data: dict[str, pd.DataFrame] = {}
        self.subscribers: list[Callable[[str, MarketData], None]] = []
        self.is_running = False
        self.update_interval = 60  # seconds

        logger.info(f"Live data manager initialized with {len(symbols)} symbols")

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        self.symbols.add(symbol)
        logger.info(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from tracking."""
        self.symbols.discard(symbol)
        if symbol in self.data_cache:
            del self.data_cache[symbol]
        logger.info(f"Removed symbol: {symbol}")

    def subscribe(self, callback: Callable[[str, MarketData], None]) -> None:
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
        logger.info("Added market data subscriber")

    def unsubscribe(self, callback: Callable[[str, MarketData], None]) -> None:
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info("Removed market data subscriber")

    async def start(self) -> None:
        """Start the data manager."""
        self.is_running = True
        logger.info("Starting live data manager")

        # Load initial historical data
        await self._load_historical_data()

        # Start the update loop
        asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop the data manager."""
        self.is_running = False
        logger.info("Stopping live data manager")

    async def _load_historical_data(self) -> None:
        """Load historical data for all symbols."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Load 30 days of data

        for symbol in self.symbols:
            try:
                bars = self.broker.get_bars(symbol, start_date, end_date, "1Day")
                if bars and hasattr(bars, "df"):
                    df = bars.df
                    if symbol in df.columns.get_level_values(0):
                        symbol_data = df[symbol]
                        self.historical_data[symbol] = symbol_data
                        logger.debug(
                            f"Loaded historical data for {symbol}: {len(symbol_data)} bars"
                        )
            except Exception as e:
                logger.warning(f"Failed to load historical data for {symbol}: {e}")

    async def _update_loop(self) -> None:
        """Main update loop for market data."""
        while self.is_running:
            try:
                await self._update_all_symbols()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _update_all_symbols(self) -> None:
        """Update market data for all symbols."""
        for symbol in self.symbols:
            try:
                await self._update_symbol(symbol)
            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")

    async def _update_symbol(self, symbol: str) -> None:
        """Update market data for a single symbol."""
        try:
            # Get latest bar from broker
            latest_bar = self.broker.get_latest_bar(symbol)
            if not latest_bar:
                return

            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                open=latest_bar["open"],
                high=latest_bar["high"],
                low=latest_bar["low"],
                close=latest_bar["close"],
                volume=latest_bar["volume"],
                timestamp=latest_bar["timestamp"],
            )

            # Update cache
            self.data_cache[symbol] = market_data

            # Update historical data
            if symbol in self.historical_data:
                new_row = pd.DataFrame([market_data.to_dict()])
                new_row.set_index("timestamp", inplace=True)
                self.historical_data[symbol] = pd.concat([self.historical_data[symbol], new_row])

                # Keep only last 1000 bars
                if len(self.historical_data[symbol]) > 1000:
                    self.historical_data[symbol] = self.historical_data[symbol].tail(1000)

            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(symbol, market_data)
                except Exception as e:
                    logger.warning(f"Subscriber callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to update symbol {symbol}: {e}")

    def get_latest_data(self, symbol: str) -> MarketData | None:
        """Get latest market data for a symbol."""
        return self.data_cache.get(symbol)

    def get_all_latest_data(self) -> dict[str, MarketData]:
        """Get latest market data for all symbols."""
        return self.data_cache.copy()

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame | None:
        """Get historical data for a symbol."""
        if symbol not in self.historical_data:
            return None

        df = self.historical_data[symbol]
        cutoff_date = datetime.now() - timedelta(days=days)
        return df[df.index >= cutoff_date]

    def get_technical_indicators(self, symbol: str, days: int = 30) -> dict[str, float] | None:
        """Calculate technical indicators for a symbol."""
        df = self.get_historical_data(symbol, days)
        if df is None or df.empty:
            return None

        try:
            # Calculate simple moving averages
            sma_20 = df["close"].rolling(window=20).mean().iloc[-1]
            sma_50 = df["close"].rolling(window=50).mean().iloc[-1]

            # Calculate ATR
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]

            # Calculate RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))

            return {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "atr": atr,
                "rsi": rsi,
                "current_price": df["close"].iloc[-1],
            }
        except Exception as e:
            logger.warning(f"Failed to calculate indicators for {symbol}: {e}")
            return None

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        # Simple check - can be enhanced with actual market hours
        return now.weekday() < 5 and 9 <= now.hour < 16
