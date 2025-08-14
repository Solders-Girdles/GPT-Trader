"""
Redundant Data Source Manager
Phase 2.5 - Day 3

Manages multiple data sources with automatic failover and quality monitoring.
"""

import asyncio
import builtins
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import aiohttp
import pandas as pd
import yfinance as yf

from ..database.database_manager import get_db_manager
from .realtime_feed import DataSource, DataValidator, MarketData

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""

    EXCELLENT = "excellent"  # Real-time, validated
    GOOD = "good"  # Near real-time, mostly complete
    FAIR = "fair"  # Delayed or partial data
    POOR = "poor"  # Significant delays or gaps
    UNAVAILABLE = "unavailable"


@dataclass
class DataSourceStatus:
    """Status of a data source"""

    source: DataSource
    is_active: bool
    quality: DataQuality
    latency_ms: float
    error_rate: float
    last_success: datetime | None
    last_error: str | None
    consecutive_errors: int
    data_points_received: int


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""

    # Timeouts
    connection_timeout: float = 5.0
    read_timeout: float = 10.0

    # Quality thresholds
    max_latency_ms: float = 1000.0
    max_error_rate: float = 0.1
    min_data_points: int = 10

    # Failover settings
    failover_threshold: int = 3  # Consecutive errors before failover
    recovery_check_interval: int = 300  # Check failed sources every 5 minutes

    # API endpoints
    alpaca_base_url: str = "https://data.alpaca.markets/v2"
    polygon_base_url: str = "https://api.polygon.io/v2"
    iex_base_url: str = "https://cloud.iexapis.com/stable"
    yahoo_base_url: str = "https://query1.finance.yahoo.com/v8/finance"


class DataSourceManager:
    """
    Manages multiple data sources with redundancy and failover.

    Features:
    - Automatic failover to backup sources
    - Data quality monitoring
    - Latency tracking
    - Error rate monitoring
    - Automatic recovery attempts
    """

    def __init__(self, config: DataSourceConfig | None = None):
        self.config = config or DataSourceConfig()
        self.validator = DataValidator()

        # Data source priority order
        self.source_priority = [
            DataSource.ALPACA,
            DataSource.POLYGON,
            DataSource.IEX,
            DataSource.YAHOO,
        ]

        # Source status tracking
        self.source_status: dict[DataSource, DataSourceStatus] = {}
        for source in self.source_priority:
            self.source_status[source] = DataSourceStatus(
                source=source,
                is_active=True,
                quality=DataQuality.GOOD,
                latency_ms=0.0,
                error_rate=0.0,
                last_success=None,
                last_error=None,
                consecutive_errors=0,
                data_points_received=0,
            )

        # Current primary source
        self.primary_source = self.source_priority[0]

        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

        # Thread pool for parallel requests
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Database manager
        self.db_manager = get_db_manager()

        # Recovery check task
        self.recovery_task = None

        logger.info(f"DataSourceManager initialized with {len(self.source_priority)} sources")

    async def fetch_market_data(
        self, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """
        Fetch market data with automatic failover.

        Args:
            symbol: Stock symbol
            timeout: Request timeout

        Returns:
            MarketData or None if all sources fail
        """
        self.request_count += 1

        for source in self._get_active_sources():
            try:
                start_time = time.time()

                # Try fetching from this source
                data = await self._fetch_from_source(source, symbol, timeout)

                if data:
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000

                    # Validate data
                    is_valid, error_msg = self.validator.validate_market_data(data)

                    if is_valid:
                        # Update source status
                        self._update_source_success(source, latency_ms)
                        self.success_count += 1
                        return data
                    else:
                        logger.warning(f"Invalid data from {source.value}: {error_msg}")
                        self._update_source_error(source, error_msg)
                else:
                    self._update_source_error(source, "No data returned")

            except TimeoutError:
                self._update_source_error(source, "Timeout")
                logger.warning(f"Timeout fetching from {source.value}")

            except Exception as e:
                self._update_source_error(source, str(e))
                logger.error(f"Error fetching from {source.value}: {e}")

        # All sources failed
        self.error_count += 1
        logger.error(f"All data sources failed for {symbol}")
        return None

    async def fetch_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """
        Fetch historical data with failover.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data or None
        """
        for source in self._get_active_sources():
            try:
                if source == DataSource.YAHOO:
                    data = await self._fetch_yahoo_historical(symbol, start_date, end_date)
                elif source == DataSource.ALPACA:
                    data = await self._fetch_alpaca_historical(symbol, start_date, end_date)
                elif source == DataSource.POLYGON:
                    data = await self._fetch_polygon_historical(symbol, start_date, end_date)
                else:
                    continue

                if data is not None and not data.empty:
                    self._update_source_success(source, 0)
                    return data

            except Exception as e:
                self._update_source_error(source, str(e))
                logger.error(f"Error fetching historical from {source.value}: {e}")

        logger.error(f"Failed to fetch historical data for {symbol}")
        return None

    async def _fetch_from_source(
        self, source: DataSource, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """Fetch data from specific source"""

        if source == DataSource.ALPACA:
            return await self._fetch_alpaca_quote(symbol, timeout)
        elif source == DataSource.POLYGON:
            return await self._fetch_polygon_quote(symbol, timeout)
        elif source == DataSource.IEX:
            return await self._fetch_iex_quote(symbol, timeout)
        elif source == DataSource.YAHOO:
            return await self._fetch_yahoo_quote(symbol, timeout)
        else:
            return None

    async def _fetch_alpaca_quote(
        self, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """Fetch quote from Alpaca API"""
        try:
            url = f"{self.config.alpaca_base_url}/stocks/{symbol}/quotes/latest"
            headers = {
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
            }

            timeout_val = timeout or self.config.read_timeout

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_val)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data.get("quote", {})

                        return MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(quote["t"]),
                            price=Decimal(str((quote["bp"] + quote["ap"]) / 2)),
                            bid=Decimal(str(quote["bp"])),
                            ask=Decimal(str(quote["ap"])),
                            bid_size=quote.get("bs"),
                            ask_size=quote.get("as"),
                            source=DataSource.ALPACA,
                        )
            return None

        except Exception as e:
            logger.error(f"Alpaca quote error: {e}")
            return None

    async def _fetch_polygon_quote(
        self, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """Fetch quote from Polygon API"""
        try:
            import os

            api_key = os.getenv("POLYGON_API_KEY", "")
            url = f"{self.config.polygon_base_url}/last/nbbo/{symbol}"
            params = {"apikey": api_key}

            timeout_val = timeout or self.config.read_timeout

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=timeout_val)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("results", {})

                        return MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(result["t"] / 1000),
                            price=Decimal(str((result["P"] + result["p"]) / 2)),
                            bid=Decimal(str(result["p"])),
                            ask=Decimal(str(result["P"])),
                            bid_size=result.get("s"),
                            ask_size=result.get("S"),
                            source=DataSource.POLYGON,
                        )
            return None

        except Exception as e:
            logger.error(f"Polygon quote error: {e}")
            return None

    async def _fetch_iex_quote(
        self, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """Fetch quote from IEX Cloud"""
        try:
            import os

            api_key = os.getenv("IEX_API_KEY", "")
            url = f"{self.config.iex_base_url}/stock/{symbol}/quote"
            params = {"token": api_key}

            timeout_val = timeout or self.config.read_timeout

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=timeout_val)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        return MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(data["latestUpdate"] / 1000),
                            price=Decimal(str(data["latestPrice"])),
                            bid=Decimal(str(data.get("iexBidPrice", data["latestPrice"]))),
                            ask=Decimal(str(data.get("iexAskPrice", data["latestPrice"]))),
                            bid_size=data.get("iexBidSize"),
                            ask_size=data.get("iexAskSize"),
                            volume=data.get("volume"),
                            source=DataSource.IEX,
                        )
            return None

        except Exception as e:
            logger.error(f"IEX quote error: {e}")
            return None

    async def _fetch_yahoo_quote(
        self, symbol: str, timeout: float | None = None
    ) -> MarketData | None:
        """Fetch quote from Yahoo Finance"""
        try:
            # Use yfinance in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def get_yahoo_data():
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if "regularMarketPrice" in info:
                    return MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=Decimal(str(info["regularMarketPrice"])),
                        bid=Decimal(str(info.get("bid", info["regularMarketPrice"]))),
                        ask=Decimal(str(info.get("ask", info["regularMarketPrice"]))),
                        volume=info.get("volume"),
                        source=DataSource.YAHOO,
                    )
                return None

            timeout_val = timeout or self.config.read_timeout
            future = loop.run_in_executor(self.executor, get_yahoo_data)

            try:
                result = await asyncio.wait_for(future, timeout=timeout_val)
                return result
            except builtins.TimeoutError:
                raise TimeoutError("Yahoo request timeout")

        except Exception as e:
            logger.error(f"Yahoo quote error: {e}")
            return None

    async def _fetch_yahoo_historical(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """Fetch historical data from Yahoo Finance"""
        try:
            loop = asyncio.get_event_loop()

            def get_historical():
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    data.columns = [col.lower() for col in data.columns]
                    return data
                return None

            future = loop.run_in_executor(self.executor, get_historical)
            result = await asyncio.wait_for(future, timeout=30.0)
            return result

        except Exception as e:
            logger.error(f"Yahoo historical error: {e}")
            return None

    async def _fetch_alpaca_historical(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """Fetch historical data from Alpaca"""
        try:
            import os

            url = f"{self.config.alpaca_base_url}/stocks/{symbol}/bars"
            headers = {
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
            }
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "timeframe": "1Day",
                "limit": 10000,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        bars = data.get("bars", [])

                        if bars:
                            df = pd.DataFrame(bars)
                            df["timestamp"] = pd.to_datetime(df["t"])
                            df.set_index("timestamp", inplace=True)
                            df.rename(
                                columns={
                                    "o": "open",
                                    "h": "high",
                                    "l": "low",
                                    "c": "close",
                                    "v": "volume",
                                },
                                inplace=True,
                            )
                            return df[["open", "high", "low", "close", "volume"]]
            return None

        except Exception as e:
            logger.error(f"Alpaca historical error: {e}")
            return None

    async def _fetch_polygon_historical(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """Fetch historical data from Polygon"""
        try:
            import os

            api_key = os.getenv("POLYGON_API_KEY", "")
            url = f"{self.config.polygon_base_url}/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": api_key, "limit": 50000}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])

                        if results:
                            df = pd.DataFrame(results)
                            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                            df.set_index("timestamp", inplace=True)
                            df.rename(
                                columns={
                                    "o": "open",
                                    "h": "high",
                                    "l": "low",
                                    "c": "close",
                                    "v": "volume",
                                },
                                inplace=True,
                            )
                            return df[["open", "high", "low", "close", "volume"]]
            return None

        except Exception as e:
            logger.error(f"Polygon historical error: {e}")
            return None

    def _get_active_sources(self) -> list[DataSource]:
        """Get list of active sources in priority order"""
        active_sources = []

        for source in self.source_priority:
            status = self.source_status[source]
            if status.is_active and status.consecutive_errors < self.config.failover_threshold:
                active_sources.append(source)

        # If no active sources, try all sources
        if not active_sources:
            logger.warning("No active sources, trying all sources")
            return self.source_priority

        return active_sources

    def _update_source_success(self, source: DataSource, latency_ms: float):
        """Update source status on successful request"""
        status = self.source_status[source]

        status.last_success = datetime.now()
        status.consecutive_errors = 0
        status.data_points_received += 1

        # Update latency (exponential moving average)
        if status.latency_ms == 0:
            status.latency_ms = latency_ms
        else:
            status.latency_ms = 0.9 * status.latency_ms + 0.1 * latency_ms

        # Update error rate
        total_requests = status.data_points_received + status.consecutive_errors
        status.error_rate = status.consecutive_errors / max(1, total_requests)

        # Update quality assessment
        status.quality = self._assess_quality(status)

        # Activate if was inactive
        if not status.is_active:
            logger.info(f"Reactivating {source.value} after successful request")
            status.is_active = True

    def _update_source_error(self, source: DataSource, error: str):
        """Update source status on error"""
        status = self.source_status[source]

        status.last_error = error
        status.consecutive_errors += 1

        # Update error rate
        total_requests = status.data_points_received + status.consecutive_errors
        status.error_rate = status.consecutive_errors / max(1, total_requests)

        # Check if should deactivate
        if status.consecutive_errors >= self.config.failover_threshold:
            if status.is_active:
                logger.warning(
                    f"Deactivating {source.value} after {status.consecutive_errors} errors"
                )
                status.is_active = False
                status.quality = DataQuality.UNAVAILABLE

                # Try next source
                self._select_new_primary()

    def _assess_quality(self, status: DataSourceStatus) -> DataQuality:
        """Assess data quality based on metrics"""

        if not status.is_active:
            return DataQuality.UNAVAILABLE

        # Check latency
        if status.latency_ms > self.config.max_latency_ms:
            return DataQuality.POOR

        # Check error rate
        if status.error_rate > self.config.max_error_rate:
            return DataQuality.POOR

        # Check data points
        if status.data_points_received < self.config.min_data_points:
            return DataQuality.FAIR

        # Assess based on latency
        if status.latency_ms < 100:
            return DataQuality.EXCELLENT
        elif status.latency_ms < 500:
            return DataQuality.GOOD
        else:
            return DataQuality.FAIR

    def _select_new_primary(self):
        """Select new primary source when current fails"""
        for source in self.source_priority:
            status = self.source_status[source]
            if status.is_active and status.quality != DataQuality.UNAVAILABLE:
                self.primary_source = source
                logger.info(f"Selected {source.value} as new primary source")
                return

        logger.error("No available data sources!")

    async def start_recovery_check(self):
        """Start periodic check to recover failed sources"""

        async def check_recovery():
            while True:
                await asyncio.sleep(self.config.recovery_check_interval)

                for source in self.source_priority:
                    status = self.source_status[source]

                    if not status.is_active:
                        logger.info(f"Attempting to recover {source.value}")

                        # Try a test request
                        try:
                            data = await self._fetch_from_source(source, "AAPL", timeout=5.0)
                            if data:
                                logger.info(f"Successfully recovered {source.value}")
                                status.is_active = True
                                status.consecutive_errors = 0
                                status.quality = DataQuality.GOOD
                        except Exception as e:
                            logger.debug(f"Recovery failed for {source.value}: {e}")

        self.recovery_task = asyncio.create_task(check_recovery())

    def get_source_status(self) -> dict[DataSource, DataSourceStatus]:
        """Get current status of all data sources"""
        return self.source_status.copy()

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics"""
        success_rate = self.success_count / max(1, self.request_count)

        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "primary_source": self.primary_source.value,
            "active_sources": len(self._get_active_sources()),
            "source_quality": {
                source.value: status.quality.value for source, status in self.source_status.items()
            },
        }

    async def shutdown(self):
        """Shutdown the manager"""
        if self.recovery_task:
            self.recovery_task.cancel()

        self.executor.shutdown(wait=True)
        logger.info("DataSourceManager shutdown complete")


# Import os for environment variables
import os
