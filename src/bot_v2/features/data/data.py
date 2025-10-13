"""
Main data management orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from collections.abc import Mapping, MutableMapping

from bot_v2.data_providers import get_data_provider
from bot_v2.utilities import (
    optional_import,
    console_data,
    console_success,
    console_error,
    console_warning,
    console_cache,
    console_storage,
    log_operation,
    get_logger,
)

from bot_v2.features.data.types import (
    DataQuery,
    StorageStats,
    DataSource,
    DataType,
)
from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.cache import DataCache
from bot_v2.features.data.quality import DataQualityChecker


# Lazy imports for heavy dependencies
pandas = optional_import("pandas")

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
else:
    pd = cast(Any, pandas)

# Global instances
_storage = DataStorage()
_cache = DataCache()
_quality_checker = DataQualityChecker()

# Logger
logger = get_logger(__name__)


def store_data(
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
    with log_operation(
        "store_data", logger, symbol=symbol, data_type=data_type.value, source=source.value
    ):
        try:
            # Check data quality
            quality = _quality_checker.check_quality(data)
            if not quality.is_acceptable():
                console_warning(
                    f"Data quality below threshold: {quality.overall_score():.2%}",
                    symbol=symbol,
                    quality_score=quality.overall_score(),
                )

            # Store data
            success = bool(
                _storage.store(
                    symbol=symbol, data=data, data_type=data_type, source=source, metadata=metadata
                )
            )

            if success:
                console_success(f"Stored {len(data)} records", symbol=symbol, records=len(data))
                logger.info(
                    "Data stored successfully",
                    symbol=symbol,
                    records=len(data),
                    data_type=data_type.value,
                    source=source.value,
                )

                # Update cache
                query = DataQuery(
                    symbols=[symbol],
                    start_date=data.index.min(),
                    end_date=data.index.max(),
                    data_type=data_type,
                    source=source,
                )
                _cache.put(query.get_cache_key(), data)
                console_cache("Updated cache", cache_key=query.get_cache_key())

            return success

        except Exception as e:
            console_error(f"Failed to store data", error=str(e), symbol=symbol)
            logger.error("Data storage failed", error=str(e), symbol=symbol, exc_info=True)
            return False


def fetch_data(query: DataQuery, use_cache: bool = True) -> pd.DataFrame | None:
    """
    Fetch data based on query.

    Args:
        query: Data query
        use_cache: Whether to check cache first

    Returns:
        DataFrame with requested data or None
    """
    with log_operation("fetch_data", logger, symbols=query.symbols, use_cache=use_cache):
        # Check cache first
        if use_cache:
            cache_key = query.get_cache_key()
            cached_data = cast(Optional[pd.DataFrame], _cache.get(cache_key))
            if cached_data is not None:
                console_cache("Cache hit", cache_key=cache_key, records=len(cached_data))
                logger.info("Cache hit", cache_key=cache_key, records=len(cached_data))
                return cached_data

        # Fetch from storage
        data = cast(Optional[pd.DataFrame], _storage.fetch(query))

        if data is not None and not data.empty:
            console_storage("Fetched from storage", records=len(data), symbols=query.symbols)
            logger.info("Data fetched from storage", records=len(data), symbols=query.symbols)

            # Update cache
            if use_cache:
                _cache.put(query.get_cache_key(), data)
                console_cache("Updated cache", cache_key=query.get_cache_key())

            return data

        # If not in storage, try downloading
        if query.source == DataSource.YAHOO or query.source is None:
            console_data("Attempting download", symbols=query.symbols, source="yahoo")
            downloaded = download_from_yahoo(
                symbols=query.symbols,
                start=query.start_date,
                end=query.end_date,
                interval=query.interval,
            )

            if downloaded:
                # Store for future use
                for symbol in query.symbols:
                    frame = downloaded.get(symbol)
                    if frame is not None and not frame.empty:
                        store_data(
                            symbol=symbol,
                            data=frame,
                            data_type=query.data_type,
                            source=DataSource.YAHOO,
                        )

                if len(query.symbols) == 1:
                    return downloaded.get(query.symbols[0])

                # Combine multi-symbol data using column MultiIndex
                combined_frames = [frame for frame in downloaded.values() if frame is not None]
                if combined_frames:
                    return pd.concat(combined_frames, axis=1, keys=downloaded.keys())

            return None

        return None


def cache_data(key: str, data: pd.DataFrame, ttl_seconds: int = 3600) -> bool:
    """
    Cache data with expiration.

    Args:
        key: Cache key
        data: Data to cache
        ttl_seconds: Time to live in seconds

    Returns:
        True if cached successfully
    """
    with log_operation("cache_data", logger, cache_key=key, ttl_seconds=ttl_seconds):
        success = bool(_cache.put(key, data, ttl_seconds))
        if success:
            console_cache("Data cached", cache_key=key, records=len(data))
        return success


def get_cache(key: str) -> pd.DataFrame | None:
    """
    Get data from cache.

    Args:
        key: Cache key

    Returns:
        Cached data or None
    """
    with log_operation("get_cache", logger, cache_key=key):
        return cast(Optional[pd.DataFrame], _cache.get(key))


def download_historical(
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
    with log_operation("download_historical", logger, symbols=symbols, source=source.value):
        results: dict[str, pd.DataFrame] = {}

        if source == DataSource.YAHOO:
            results = download_from_yahoo(symbols, start_date, end_date, interval)
        else:
            console_warning(f"Source {source.value} not implemented", source=source.value)
            logger.warning("Unsupported data source", source=source.value)

        # Store downloaded data
        for symbol, frame in results.items():
            if not frame.empty:
                store_data(symbol=symbol, data=frame, data_type=DataType.OHLCV, source=source)

        return results


def clean_old_data(days_to_keep: int = 365) -> int:
    """
    Clean data older than specified days.

    Args:
        days_to_keep: Number of days to keep

    Returns:
        Number of records deleted
    """
    with log_operation("clean_old_data", logger, days_to_keep=days_to_keep):
        cutoff = datetime.now() - timedelta(days=days_to_keep)

        deleted = int(_storage.delete_before(cutoff))
        _cache.clear_expired()

        console_storage("Cleaned old records", records_deleted=deleted, days_to_keep=days_to_keep)
        logger.info("Old data cleaned", records_deleted=deleted, days_to_keep=days_to_keep)
        return deleted


def get_storage_stats() -> StorageStats:
    """
    Get storage statistics.

    Returns:
        StorageStats object
    """
    storage_stats = cast(dict[str, Any], _storage.get_stats())
    cache_stats = cast(dict[str, Any], _cache.get_stats())

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


def download_from_yahoo(
    symbols: list[str], start: datetime, end: datetime, interval: str = "1d"
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
    with log_operation("download_from_yahoo", logger, symbols=symbols, interval=interval):
        results: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            with log_operation("download_symbol", logger, symbol=symbol):
                try:
                    console_data("Downloading", symbol=symbol, source="yahoo")
                    provider = get_data_provider()
                    period_days = max((end - start).days, 1)
                    raw = cast(
                        pd.DataFrame,
                        provider.get_historical_data(symbol, period=f"{period_days}d"),
                    )

                    if raw.empty:
                        console_warning("No data available", symbol=symbol)
                        logger.warning("No data available for symbol", symbol=symbol)
                        continue

                    raw.index = pd.to_datetime(raw.index)
                    idx_array = raw.index.to_pydatetime()
                    mask = (idx_array >= start) & (idx_array <= end)
                    data = raw.loc[mask].copy()

                    if not data.empty:
                        # Standardize columns
                        data.columns = data.columns.str.lower()
                        results[symbol] = data
                        console_success("Downloaded", symbol=symbol, records=len(data))
                        logger.info(
                            "Data downloaded successfully", symbol=symbol, records=len(data)
                        )
                    else:
                        console_warning("No data available", symbol=symbol)
                        logger.warning("No data available for symbol", symbol=symbol)

                except Exception as e:
                    console_error("Download failed", symbol=symbol, error=str(e))
                    logger.error("Data download failed", symbol=symbol, error=str(e), exc_info=True)

        return results


def export_data(query: DataQuery, format: str = "csv", path: str = "./exports") -> bool:
    """
    Export data to file.

    Args:
        query: Data query
        format: Export format ('csv', 'json', 'parquet')
        path: Export path

    Returns:
        True if exported successfully
    """
    with log_operation("export_data", logger, format=format, path=path):
        data = fetch_data(query)

        if data is None or data.empty:
            console_warning("No data to export", query=query.get_cache_key())
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
                console_error("Unsupported export format", format=format)
                return False

            console_success("Data exported", filepath=filepath, format=format, records=len(data))
            logger.info(
                "Data exported successfully", filepath=filepath, format=format, records=len(data)
            )
            return True

        except Exception as e:
            console_error("Export failed", filepath=filepath, error=str(e))
            logger.error("Data export failed", filepath=filepath, error=str(e), exc_info=True)
            return False


def import_data(
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
    with log_operation(
        "import_data", logger, filepath=filepath, symbol=symbol, source=source.value
    ):
        try:
            # Determine format from extension
            ext = os.path.splitext(filepath)[1].lower()

            if ext == ".csv":
                if not pandas.is_available():
                    console_error("pandas not available for CSV import")
                    return False
                data = cast(pd.DataFrame, pandas.read_csv(filepath, index_col=0, parse_dates=True))
            elif ext == ".json":
                if not pandas.is_available():
                    console_error("pandas not available for JSON import")
                    return False
                data = cast(pd.DataFrame, pandas.read_json(filepath))
            elif ext == ".parquet":
                if not pandas.is_available():
                    console_error("pandas not available for Parquet import")
                    return False
                data = cast(pd.DataFrame, pandas.read_parquet(filepath))
            else:
                console_error("Unsupported file format", format=ext)
                return False

            console_success("Data imported", filepath=filepath, symbol=symbol, records=len(data))
            logger.info(
                "Data imported successfully", filepath=filepath, symbol=symbol, records=len(data)
            )

            # Store imported data
            return store_data(symbol=symbol, data=data, data_type=data_type, source=source)

        except Exception as e:
            console_error("Import failed", filepath=filepath, error=str(e))
            logger.error("Data import failed", filepath=filepath, error=str(e), exc_info=True)
            return False
