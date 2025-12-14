"""
HTTP Response Cache with endpoint-specific TTLs.

Provides thread-safe caching for GET requests to reduce API calls
for stable data like products and accounts.
"""

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="response_cache")


@dataclass
class CachedResponse:
    """A cached HTTP response with metadata."""

    data: dict[str, Any]
    timestamp: float
    endpoint: str
    hits: int = 0


@dataclass
class ResponseCache:
    """Thread-safe HTTP response cache with endpoint-specific TTLs.

    Caches GET responses based on the full path (including query params).
    Different endpoints have different TTL values based on data volatility.

    Example usage:
        cache = ResponseCache()
        cached = cache.get("/api/v3/brokerage/products")
        if cached is None:
            response = client.get("/api/v3/brokerage/products")
            cache.set("/api/v3/brokerage/products", response)
    """

    # Endpoint pattern -> TTL in seconds
    # More specific patterns should come first
    ENDPOINT_TTLS: dict[str, float] = field(default_factory=lambda: {
        # Market data (very volatile)
        "**/best_bid_ask*": 5.0,
        "**/ticker*": 5.0,
        "**/market/**": 10.0,

        # Position/balance data (moderate volatility)
        "**/positions*": 30.0,
        "**/cfm/positions*": 30.0,
        "**/intx/positions*": 30.0,
        "**/accounts*": 60.0,
        "**/cfm/balance_summary*": 60.0,

        # Order data (changes frequently during trading)
        "**/orders*": 15.0,
        "**/fills*": 30.0,

        # Product metadata (stable, rarely changes)
        "**/products*": 300.0,
        "**/product*": 300.0,
    })

    default_ttl: float = 30.0
    max_size: int = 1000
    enabled: bool = True

    _cache: dict[str, CachedResponse] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"hits": 0, "misses": 0, "evictions": 0})

    def _get_ttl_for_endpoint(self, path: str) -> float:
        """Get TTL for a given endpoint path using pattern matching."""
        for pattern, ttl in self.ENDPOINT_TTLS.items():
            if fnmatch.fnmatch(path, pattern):
                return ttl
        return self.default_ttl

    def _make_cache_key(self, path: str) -> str:
        """Create a cache key from the request path."""
        # Normalize path - remove leading/trailing slashes, lowercase
        return path.strip("/").lower()

    def _is_expired(self, entry: CachedResponse, path: str) -> bool:
        """Check if a cache entry has expired."""
        ttl = self._get_ttl_for_endpoint(path)
        return (time.time() - entry.timestamp) > ttl

    def _evict_oldest(self) -> None:
        """Evict the oldest entry when cache is full."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1

    def get(self, path: str) -> dict[str, Any] | None:
        """Get a cached response if it exists and is not expired.

        Args:
            path: The request path (e.g., "/api/v3/brokerage/products")

        Returns:
            The cached response data, or None if not cached or expired.
        """
        if not self.enabled:
            return None

        cache_key = self._make_cache_key(path)

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if self._is_expired(entry, path):
                # Remove expired entry
                del self._cache[cache_key]
                self._stats["misses"] += 1
                return None

            entry.hits += 1
            self._stats["hits"] += 1
            return entry.data

    def set(self, path: str, data: dict[str, Any]) -> None:
        """Cache a response.

        Args:
            path: The request path
            data: The response data to cache
        """
        if not self.enabled:
            return

        cache_key = self._make_cache_key(path)

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_oldest()

            self._cache[cache_key] = CachedResponse(
                data=data,
                timestamp=time.time(),
                endpoint=path,
            )

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Use after mutations (POST, DELETE) to ensure stale data is cleared.

        Args:
            pattern: Glob pattern to match (e.g., "**/orders*" or "**/accounts*")

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(f"/{key}", pattern)
            ]

            for key in keys_to_remove:
                del self._cache[key]

            if keys_to_remove:
                logger.debug(f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")

            return len(keys_to_remove)

    def invalidate_all(self) -> int:
        """Clear the entire cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"Cleared entire cache ({count} entries)")
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, size, and evictions.
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "evictions": self._stats["evictions"],
                "enabled": self.enabled,
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}
