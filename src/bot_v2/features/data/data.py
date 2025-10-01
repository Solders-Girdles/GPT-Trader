"""
Main data management orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from bot_v2.data_providers import get_data_provider
from bot_v2.features.data.cache import DataCache
from bot_v2.features.data.quality import DataQualityChecker
from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.types import (
    DataQuery,
    DataSource,
    DataType,
    StorageStats,
)

logger = logging.getLogger(__name__)


class DataService:
    """
    Data management service with dependency injection.

    Manages storage, caching, and quality checking for market data.
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        cache: Optional[DataCache] = None,
        quality_checker: Optional[DataQualityChecker] = None,
    ):
        """
        Initialize data service with injected dependencies.

        Args:
            storage: Data storage backend (creates default if None)
            cache: Cache backend (creates default if None)
            quality_checker: Quality checker (creates default if None)
        """
        self.storage = storage or DataStorage()
        self.cache = cache or DataCache()
        self.quality_checker = quality_checker or DataQualityChecker()

    def store_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_type: DataType = DataType.OHLCV,
        source: DataSource = DataSource.YAHOO,
        metadata: dict | None = None,
    ) -> bool:
        """
        Store data to persistent storage.

        Args:
            symbol: Stock symbol
            data: DataFrame to store
            data_type: Type of data
            source: Data source
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        try:
            # Check data quality
            quality = self.quality_checker.check_quality(data)
            if not quality.is_acceptable():
                logger.warning(
                    "Data quality below threshold",
                    extra={
                        "symbol": symbol,
                        "quality_score": quality.overall_score(),
                        "threshold": 0.7,
                    },
                )

            # Store data
            success = self.storage.store(
                symbol=symbol, data=data, data_type=data_type, source=source, metadata=metadata
            )

            if success:
                logger.info(
                    "Stored data successfully",
                    extra={"symbol": symbol, "records_count": len(data)},
                )

                # Update cache
                query = DataQuery(
                    symbols=[symbol],
                    start_date=data.index.min(),
                    end_date=data.index.max(),
                    data_type=data_type,
                    source=source,
                )
                self.cache.put(query.get_cache_key(), data)

            return success

        except Exception as e:
            logger.error(
                "Failed to store data",
                extra={"symbol": symbol, "error": str(e)},
                exc_info=True,
            )
            return False


    def fetch_data(self, query: DataQuery, use_cache: bool = True) -> pd.DataFrame | None:
        """
        Fetch data based on query.

        Args:
            query: Data query
            use_cache: Whether to check cache first

        Returns:
            DataFrame with requested data or None
        """
        # Check cache first
        if use_cache:
            cache_key = query.get_cache_key()
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug("Cache hit", extra={"cache_key": cache_key})
                return cached_data

        # Fetch from storage
        data = self.storage.fetch(query)

        if data is not None and not data.empty:
            logger.info(
                "Fetched data from storage",
                extra={"records_count": len(data), "symbols": query.symbols},
            )

            # Update cache
            if use_cache:
                self.cache.put(query.get_cache_key(), data)

            return data

        # If not in storage, try downloading
        if query.source == DataSource.YAHOO or query.source is None:
            data = self._download_from_yahoo(
                symbols=query.symbols,
                start=query.start_date,
                end=query.end_date,
                interval=query.interval,
            )

            if data is not None:
                # Store for future use
                for symbol in query.symbols:
                    if symbol in data:
                        self.store_data(
                            symbol=symbol,
                            data=data[symbol],
                            data_type=query.data_type,
                            source=DataSource.YAHOO,
                        )

                return data

        return None


    def cache_data(self, key: str, data: pd.DataFrame, ttl_seconds: int = 3600) -> bool:
        """
        Cache data with expiration.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if cached successfully
        """
        return self.cache.put(key, data, ttl_seconds)

    def get_cache(self, key: str) -> pd.DataFrame | None:
        """
        Get data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        return self.cache.get(key)


    def download_historical(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        source: DataSource = DataSource.YAHOO,
    ) -> dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            source: Data source

        Returns:
            Dict of symbol -> DataFrame
        """
        results = {}

        if source == DataSource.YAHOO:
            results = self._download_from_yahoo(symbols, start_date, end_date, interval)
        else:
            logger.warning("Data source not implemented", extra={"source": source.value})

        # Store downloaded data
        for symbol, data in results.items():
            if data is not None and not data.empty:
                self.store_data(symbol=symbol, data=data, data_type=DataType.OHLCV, source=source)

        return results


    def clean_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clean data older than specified days.

        Args:
            days_to_keep: Number of days to keep

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)

        deleted = self.storage.delete_before(cutoff)
        self.cache.clear_expired()

        logger.info("Cleaned old records", extra={"records_deleted": deleted, "days_to_keep": days_to_keep})
        return deleted

    def get_storage_stats(self) -> StorageStats:
        """
        Get storage statistics.

        Returns:
            StorageStats object
        """
        storage_stats = self.storage.get_stats()
        cache_stats = self.cache.get_stats()

        return StorageStats(
            total_records=storage_stats["total_records"],
            total_size_mb=storage_stats["total_size_mb"],
            oldest_record=storage_stats["oldest_record"],
            newest_record=storage_stats["newest_record"],
            symbols_count=storage_stats["symbols_count"],
            cache_entries=cache_stats["entries"],
            cache_size_mb=cache_stats["size_mb"],
            cache_hit_rate=cache_stats["hit_rate"],
        )


    def _download_from_yahoo(
        self, symbols: list[str], start: datetime, end: datetime, interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Download data from Yahoo Finance.

        Args:
            symbols: List of symbols
            start: Start date
            end: End date
            interval: Data interval

        Returns:
            Dict of symbol -> DataFrame
        """
        results = {}

        for symbol in symbols:
            try:
                logger.info("Downloading data from provider", extra={"symbol": symbol})
                provider = get_data_provider()
                data = provider.get_historical_data(
                    symbol,
                    start=start.strftime("%Y-%m-%d") if start else None,
                    end=end.strftime("%Y-%m-%d") if end else None,
                )

                if not data.empty:
                    # Standardize columns
                    data.columns = data.columns.str.lower()
                    results[symbol] = data
                    logger.info(
                        "Downloaded data successfully",
                        extra={"symbol": symbol, "records_count": len(data)},
                    )
                else:
                    logger.warning("No data available", extra={"symbol": symbol})

            except Exception as e:
                logger.error(
                    "Failed to download data",
                    extra={"symbol": symbol, "error": str(e)},
                    exc_info=True,
                )

        return results


    def export_data(self, query: DataQuery, format: str = "csv", path: str = "./exports") -> bool:
        """
        Export data to file.

        Args:
            query: Data query
            format: Export format ('csv', 'json', 'parquet')
            path: Export path

        Returns:
            True if exported successfully
        """
        data = self.fetch_data(query)

        if data is None or data.empty:
            logger.warning("No data to export", extra={"query": str(query)})
            return False

        os.makedirs(path, exist_ok=True)

        filename = f"{query.get_cache_key()}.{format}"
        filepath = os.path.join(path, filename)

        try:
            if format == "csv":
                data.to_csv(filepath)
            elif format == "json":
                data.to_json(filepath)
            elif format == "parquet":
                data.to_parquet(filepath)
            else:
                logger.error("Unsupported export format", extra={"format": format})
                return False

            logger.info("Exported data successfully", extra={"filepath": filepath, "format": format})
            return True

        except Exception as e:
            logger.error(
                "Export failed",
                extra={"filepath": filepath, "error": str(e)},
                exc_info=True,
            )
            return False


    def import_data(
        self,
        filepath: str,
        symbol: str,
        data_type: DataType = DataType.OHLCV,
        source: DataSource = DataSource.CSV,
    ) -> bool:
        """
        Import data from file.

        Args:
            filepath: Path to file
            symbol: Stock symbol
            data_type: Type of data
            source: Data source

        Returns:
            True if imported successfully
        """
        try:
            # Determine format from extension
            ext = os.path.splitext(filepath)[1].lower()

            if ext == ".csv":
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif ext == ".json":
                data = pd.read_json(filepath)
            elif ext == ".parquet":
                data = pd.read_parquet(filepath)
            else:
                logger.error("Unsupported file format", extra={"extension": ext})
                return False

            # Store imported data
            return self.store_data(symbol=symbol, data=data, data_type=data_type, source=source)

        except Exception as e:
            logger.error(
                "Import failed",
                extra={"filepath": filepath, "error": str(e)},
                exc_info=True,
            )
            return False
