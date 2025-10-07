"""
Redis State Repository - HOT tier storage implementation

Provides sub-second access times for frequently accessed data.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

from bot_v2.state.utils.adapters import RedisAdapter

logger = logging.getLogger(__name__)

__all__ = ["RedisStateRepository"]


class RedisStateRepository:
    """
    Redis repository for HOT tier state storage.

    Provides sub-second access times for frequently accessed data.
    """

    def __init__(
        self,
        adapter: RedisAdapter,
        default_ttl: int = 3600,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        """
        Initialize Redis repository.

        Args:
            adapter: Redis adapter instance
            default_ttl: Default TTL for Redis keys in seconds
            metrics_collector: Optional metrics collector for telemetry
        """
        self.adapter = adapter
        self.default_ttl = default_ttl
        self.metrics_collector = metrics_collector

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from Redis.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.redis.operations.fetch_total")

        try:
            value = self.adapter.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Redis fetch failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )

        return None

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Store state in Redis with TTL.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata dict containing optional 'ttl_seconds'

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.redis.operations.store_total")

        try:
            ttl = metadata.get("ttl_seconds", self.default_ttl)
            return self.adapter.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Redis store failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from Redis.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.redis.operations.delete_total")

        try:
            self.adapter.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return False

    async def keys(self, pattern: str) -> list[str]:
        """
        Get Redis keys matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            List of matching keys
        """
        try:
            return self.adapter.keys(pattern)
        except Exception as e:
            logger.debug(f"Redis key lookup failed for pattern {pattern}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return []

    async def stats(self) -> dict[str, Any]:
        """
        Get Redis storage statistics.

        Returns:
            Dict containing 'key_count' and other stats
        """
        try:
            return {"key_count": self.adapter.dbsize()}
        except Exception as e:
            logger.debug(f"Redis stats collection failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return {"key_count": 0}

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items in Redis with TTL.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        if not items:
            return set()

        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.redis.operations.store_many_total"
            )
            self.metrics_collector.record_histogram(
                "state.repository.redis.operations.batch_size", float(len(items))
            )

        try:
            # Group by TTL for efficient pipeline execution
            ttl_groups: dict[int, dict[str, str]] = {}

            for key, (value, metadata) in items.items():
                ttl = metadata.get("ttl_seconds", self.default_ttl)
                if ttl not in ttl_groups:
                    ttl_groups[ttl] = {}
                ttl_groups[ttl][key] = value

            # Execute batch sets grouped by TTL
            for ttl, mapping in ttl_groups.items():
                self.adapter.msetex(mapping, ttl)

            # All items succeeded (Redis msetex is atomic per TTL group)
            return set(items.keys())
        except Exception as e:
            logger.error(f"Redis store_many failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return set()

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from Redis.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        if not keys:
            return 0

        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.redis.operations.delete_many_total"
            )
            self.metrics_collector.record_histogram(
                "state.repository.redis.operations.batch_size", float(len(keys))
            )

        try:
            return self.adapter.delete_many(keys)
        except Exception as e:
            logger.error(f"Redis delete_many failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.redis.operations.errors_total"
                )
            return 0
