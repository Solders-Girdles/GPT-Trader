"""
Main data management orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import pickle
import json
import os
from bot_v2.data_providers import get_data_provider

from bot_v2.features.data.types import (
    DataRecord,
    DataQuery,
    CacheEntry,
    StorageStats,
    DataSource,
    DataType,
    DataQuality,
    DataUpdate,
)
from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.cache import DataCache
from bot_v2.features.data.quality import DataQualityChecker


# Global instances
_storage = DataStorage()
_cache = DataCache()
_quality_checker = DataQualityChecker()


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
    try:
        # Check data quality
        quality = _quality_checker.check_quality(data)
        if not quality.is_acceptable():
            print(f"⚠️ Data quality below threshold: {quality.overall_score():.2%}")

        # Store data
        success = _storage.store(
            symbol=symbol, data=data, data_type=data_type, source=source, metadata=metadata
        )

        if success:
            print(f"✅ Stored {len(data)} records for {symbol}")

            # Update cache
            query = DataQuery(
                symbols=[symbol],
                start_date=data.index.min(),
                end_date=data.index.max(),
                data_type=data_type,
                source=source,
            )
            _cache.put(query.get_cache_key(), data)

        return success

    except Exception as e:
        print(f"❌ Failed to store data: {e}")
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
    # Check cache first
    if use_cache:
        cache_key = query.get_cache_key()
        cached_data = _cache.get(cache_key)
        if cached_data is not None:
            print(f"📦 Cache hit for {cache_key}")
            return cached_data

    # Fetch from storage
    data = _storage.fetch(query)

    if data is not None and not data.empty:
        print(f"💾 Fetched {len(data)} records from storage")

        # Update cache
        if use_cache:
            _cache.put(query.get_cache_key(), data)

        return data

    # If not in storage, try downloading
    if query.source == DataSource.YAHOO or query.source is None:
        data = download_from_yahoo(
            symbols=query.symbols,
            start=query.start_date,
            end=query.end_date,
            interval=query.interval,
        )

        if data is not None:
            # Store for future use
            for symbol in query.symbols:
                if symbol in data:
                    store_data(
                        symbol=symbol,
                        data=data[symbol],
                        data_type=query.data_type,
                        source=DataSource.YAHOO,
                    )

            return data

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
    return _cache.put(key, data, ttl_seconds)


def get_cache(key: str) -> pd.DataFrame | None:
    """
    Get data from cache.

    Args:
        key: Cache key

    Returns:
        Cached data or None
    """
    return _cache.get(key)


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
    results = {}

    if source == DataSource.YAHOO:
        results = download_from_yahoo(symbols, start_date, end_date, interval)
    else:
        print(f"⚠️ Source {source.value} not implemented")

    # Store downloaded data
    for symbol, data in results.items():
        if data is not None and not data.empty:
            store_data(symbol=symbol, data=data, data_type=DataType.OHLCV, source=source)

    return results


def clean_old_data(days_to_keep: int = 365) -> int:
    """
    Clean data older than specified days.

    Args:
        days_to_keep: Number of days to keep

    Returns:
        Number of records deleted
    """
    cutoff = datetime.now() - timedelta(days=days_to_keep)

    deleted = _storage.delete_before(cutoff)
    _cache.clear_expired()

    print(f"🧹 Cleaned {deleted} old records")
    return deleted


def get_storage_stats() -> StorageStats:
    """
    Get storage statistics.

    Returns:
        StorageStats object
    """
    storage_stats = _storage.get_stats()
    cache_stats = _cache.get_stats()

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
    results = {}

    for symbol in symbols:
        try:
            print(f"📥 Downloading {symbol} using data provider...")
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
                print(f"✅ Downloaded {len(data)} records for {symbol}")
            else:
                print(f"⚠️ No data available for {symbol}")

        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")

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
    data = fetch_data(query)

    if data is None or data.empty:
        print("No data to export")
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
            print(f"Unsupported format: {format}")
            return False

        print(f"✅ Exported to {filepath}")
        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
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
            print(f"Unsupported file format: {ext}")
            return False

        # Store imported data
        return store_data(symbol=symbol, data=data, data_type=data_type, source=source)

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
