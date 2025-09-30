"""
Local data caching implementation.

Complete isolation - no external dependencies.
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import sys
from bot_v2.features.data.types import CacheEntry


class DataCache:
    """In-memory data cache with TTL."""

    def __init__(self, max_size_mb: float = 100.0):
        """
        Initialize data cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.cache: dict[str, CacheEntry] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.total_hits = 0
        self.total_misses = 0

    def put(self, key: str, data: pd.DataFrame, ttl_seconds: int = 3600) -> bool:
        """
        Add data to cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if cached successfully
        """
        try:
            # Calculate data size
            size_bytes = data.memory_usage(deep=True).sum()

            # Check if adding would exceed limit
            current_size = self._get_total_size()
            if current_size + size_bytes > self.max_size_bytes:
                # Evict old entries
                self._evict_lru(size_bytes)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data.copy(),
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
                hit_count=0,
                size_bytes=size_bytes,
            )

            self.cache[key] = entry
            return True

        except Exception as e:
            print(f"Cache error: {e}")
            return False

    def get(self, key: str) -> pd.DataFrame | None:
        """
        Get data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        if key not in self.cache:
            self.total_misses += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.total_misses += 1
            return None

        # Update hit count
        entry.hit_count += 1
        self.total_hits += 1

        return entry.data.copy()

    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry.

        Args:
            key: Cache key

        Returns:
            True if invalidated
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.total_hits = 0
        self.total_misses = 0

    def clear_expired(self) -> int:
        """
        Clear expired entries.

        Returns:
            Number of entries cleared
        """
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = self._get_total_size()
        hit_rate = (
            self.total_hits / (self.total_hits + self.total_misses)
            if (self.total_hits + self.total_misses) > 0
            else 0
        )

        return {
            "entries": len(self.cache),
            "size_mb": total_size / (1024 * 1024),
            "hit_rate": hit_rate,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "expired_entries": sum(1 for e in self.cache.values() if e.is_expired()),
        }

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry.size_bytes for entry in self.cache.values())

    def _evict_lru(self, required_space: int):
        """
        Evict least recently used entries.

        Args:
            required_space: Space needed in bytes
        """
        # Sort by hit count and age
        sorted_entries = sorted(
            self.cache.items(), key=lambda x: (x[1].hit_count, -x[1].age_seconds())
        )

        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break

            freed_space += entry.size_bytes
            del self.cache[key]

    def warm_up(self, queries: list):
        """
        Warm up cache with common queries.

        Args:
            queries: List of queries to pre-cache
        """
        print("Warming up cache...")
        for query in queries:
            # This would fetch and cache data
            pass
