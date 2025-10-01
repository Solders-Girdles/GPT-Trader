"""
Data management feature slice - storage, caching, and historical data.

Complete isolation - no external dependencies.
"""

from bot_v2.features.data.data import DataService
from bot_v2.features.data.types import CacheEntry, DataQuery, DataRecord, StorageStats

__all__ = [
    "DataService",
    "DataRecord",
    "DataQuery",
    "CacheEntry",
    "StorageStats",
]
