"""
State Cache Manager

Manages local caching, metadata tracking, and access history
for the state management system.
"""

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.state.state_manager import StateCategory, StateMetadata


class StateCacheManager:
    """
    Manages local cache, metadata, and access patterns for state data.

    Provides fast local access and tracks state metadata for tier management.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize cache manager.

        Args:
            config: State config object with cache_size_mb attribute
        """
        self.config = config
        self._local_cache: dict[str, Any] = {}
        self._metadata_cache: dict[str, Any] = {}  # dict[str, StateMetadata]
        self._access_history: dict[str, list[datetime]] = {}

    def get(self, key: str) -> Any | None:
        """
        Get value from local cache.

        Args:
            key: State key

        Returns:
            Cached value or None if not found
        """
        if key in self._local_cache:
            self.update_access_history(key)
            return self._local_cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Store value in local cache.

        Args:
            key: State key
            value: Value to cache
        """
        self._local_cache[key] = value
        self.manage_cache_size()

    def delete(self, key: str) -> None:
        """
        Remove key from cache and metadata.

        Args:
            key: State key to remove
        """
        self._local_cache.pop(key, None)
        self._metadata_cache.pop(key, None)
        self._access_history.pop(key, None)

    def contains(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: State key

        Returns:
            True if key is cached, False otherwise
        """
        return key in self._local_cache

    def update_metadata(
        self,
        key: str,
        category: "StateCategory",
        size_bytes: int,
        checksum: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Update metadata for a state key.

        Args:
            key: State key
            category: Storage tier category
            size_bytes: Size of serialized data in bytes
            checksum: Data checksum
            ttl_seconds: Optional TTL in seconds
        """
        from bot_v2.state.state_manager import StateMetadata

        metadata = StateMetadata(
            key=key,
            category=category,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            size_bytes=size_bytes,
            checksum=checksum,
            version=1,
            ttl_seconds=ttl_seconds,
        )
        self._metadata_cache[key] = metadata

    def get_metadata(self, key: str) -> "StateMetadata | None":
        """
        Get metadata for a state key.

        Args:
            key: State key

        Returns:
            Metadata or None if not found
        """
        return self._metadata_cache.get(key)

    def update_access_history(self, key: str) -> None:
        """
        Update access history for tier management.

        Args:
            key: State key that was accessed
        """
        if key not in self._access_history:
            self._access_history[key] = []

        self._access_history[key].append(datetime.utcnow())

        # Keep only last 100 accesses
        if len(self._access_history[key]) > 100:
            self._access_history[key] = self._access_history[key][-100:]

    def get_access_history(self, key: str) -> list[datetime]:
        """
        Get access history for a key.

        Args:
            key: State key

        Returns:
            List of access timestamps
        """
        return self._access_history.get(key, [])

    def manage_cache_size(self) -> None:
        """Manage local cache size by evicting least recently accessed items."""
        max_cache_size = self.config.cache_size_mb * 1024 * 1024  # Convert to bytes
        current_size = sum(
            len(json.dumps(v, default=str).encode()) for v in self._local_cache.values()
        )

        if current_size > max_cache_size:
            # Remove least recently accessed items
            sorted_keys = sorted(
                self._local_cache.keys(),
                key=lambda k: self._access_history.get(k, [datetime.min])[-1],
            )

            while current_size > max_cache_size and sorted_keys:
                key_to_remove = sorted_keys.pop(0)
                removed_value = self._local_cache.pop(key_to_remove, None)
                if removed_value:
                    current_size -= len(json.dumps(removed_value, default=str).encode())

    def calculate_checksum(self, data: str) -> str:
        """
        Calculate SHA256 checksum for data integrity.

        Args:
            data: Data to checksum

        Returns:
            SHA256 hex digest
        """
        return hashlib.sha256(data.encode()).hexdigest()

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict containing cache stats
        """
        cache_size_bytes = sum(
            len(json.dumps(v, default=str).encode()) for v in self._local_cache.values()
        )
        return {
            "cache_keys": len(self._local_cache),
            "cache_size_bytes": cache_size_bytes,
        }
