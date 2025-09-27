"""
Cache Manager for Bot V2 Orchestration System

Simple TTL-based cache with thread safety for orchestration components.
Provides get/set operations with automatic expiration and statistics tracking.
"""

import time
import threading
from typing import Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expiry: float


class CacheManager:
    """
    Simple TTL-based cache manager with thread safety.
    
    Features:
    - Automatic expiration based on TTL
    - Thread-safe operations using locks
    - Cache statistics tracking
    - Configurable default TTL
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache manager.
        
        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry.expiry:
                    self._hits += 1
                    return entry.value
                else:
                    # Entry expired, remove it
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl or self._default_ttl
        expiry_time = time.time() + ttl
        
        with self._lock:
            self._cache[key] = CacheEntry(
                value=value,
                expiry=expiry_time
            )
    
    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of expired entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time >= entry.expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache size, hits, misses, and hit rate
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total_requests,
            'hit_rate': round(hit_rate, 3),
            'default_ttl': self._default_ttl
        }
    
    def contains(self, key: str) -> bool:
        """
        Check if key exists in cache (without affecting statistics).
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and not expired, False otherwise
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry.expiry:
                    return True
                else:
                    # Entry expired, remove it
                    del self._cache[key]
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete specific key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False


# Global cache instance for orchestration system
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Get singleton cache manager instance.
    
    Returns:
        Global CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_cache() -> None:
    """Clear the global cache."""
    cache = get_cache_manager()
    cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = get_cache_manager()
    return cache.get_stats()