"""
Data management feature slice - storage, caching, and historical data.

Complete isolation - no external dependencies.
"""

from bot_v2.features.data.data import (
    store_data,
    fetch_data,
    cache_data,
    get_cache,
    download_historical,
    clean_old_data,
    get_storage_stats,
)
from bot_v2.features.data.types import DataRecord, DataQuery, CacheEntry, StorageStats

__all__ = [
    "store_data",
    "fetch_data",
    "cache_data",
    "get_cache",
    "download_historical",
    "clean_old_data",
    "get_storage_stats",
    "DataRecord",
    "DataQuery",
    "CacheEntry",
    "StorageStats",
]
