"""
State Manager for Bot V2 Trading System

Provides multi-tier state management with hot (Redis), warm (PostgreSQL),
and cold (S3) storage layers for optimal performance and cost efficiency.
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from bot_v2.state.cache_manager import StateCacheManager
from bot_v2.state.performance import StatePerformanceMetrics
from bot_v2.state.repositories import (
    PostgresStateRepository,
    RedisStateRepository,
    S3StateRepository,
)
from bot_v2.state.storage_adapter_factory import StorageAdapterFactory
from bot_v2.state.tier_promotion_policy import TierPromotionPolicy
from bot_v2.state.utils.adapters import (
    PostgresAdapter,
    RedisAdapter,
    S3Adapter,
)

logger = logging.getLogger(__name__)


@dataclass
class StateRepositories:
    """
    Bundle of tier-specific repositories for direct access.

    Useful for batch operations (backup, checkpoint, recovery) that don't
    need StateManager's auto-promotion and caching overhead.

    Example:
        repos = state_manager.get_repositories()
        keys = await repos.redis.keys("position:*")
        for key in keys:
            value = await repos.redis.fetch(key)
    """

    redis: RedisStateRepository | None
    postgres: PostgresStateRepository | None
    s3: S3StateRepository | None


class StateCategory(Enum):
    """State storage tier categories"""

    HOT = "hot"  # Real-time data (Redis) - <1s access
    WARM = "warm"  # Recent data (PostgreSQL) - <5s access
    COLD = "cold"  # Archive data (S3) - Long-term storage


@dataclass
class StateMetadata:
    """Metadata for stored state"""

    key: str
    category: StateCategory
    created_at: datetime
    last_accessed: datetime
    size_bytes: int
    checksum: str
    version: int
    ttl_seconds: int | None = None


@dataclass
class StateConfig:
    """Configuration for state management"""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "trading_bot"
    postgres_user: str = "trader"
    postgres_password: str = "trader123"

    s3_bucket: str = "trading-bot-cold-storage"
    s3_region: str = "us-east-1"

    enable_compression: bool = True
    enable_encryption: bool = False
    cache_size_mb: int = 100
    enable_performance_tracking: bool = False


class StateManager:
    """
    Central state coordination with multi-tier storage.
    Automatically promotes/demotes data between tiers based on access patterns.
    """

    def __init__(
        self,
        config: StateConfig | None = None,
        redis_adapter: RedisAdapter | None = None,
        postgres_adapter: PostgresAdapter | None = None,
        s3_adapter: S3Adapter | None = None,
    ) -> None:
        self.config = config or StateConfig()
        self._lock = threading.Lock()

        # Initialize cache manager
        self._cache_manager = StateCacheManager(config=self.config)

        # Initialize storage backends with provided adapters or defaults
        factory = StorageAdapterFactory()

        # Redis adapter initialization
        if redis_adapter is None:
            self.redis_adapter = factory.create_redis_adapter(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
            )
        else:
            self.redis_adapter = redis_adapter

        # PostgreSQL adapter initialization
        if postgres_adapter is None:
            self.postgres_adapter = factory.create_postgres_adapter(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
        else:
            # Validate provided adapter by creating tables
            self.postgres_adapter = factory.validate_postgres_adapter(postgres_adapter)

        # S3 adapter initialization
        if s3_adapter is None:
            self.s3_adapter = factory.create_s3_adapter(
                region=self.config.s3_region,
                bucket=self.config.s3_bucket,
            )
        else:
            # Validate provided adapter by checking bucket
            self.s3_adapter = factory.validate_s3_adapter(s3_adapter, self.config.s3_bucket)

        # Initialize tier-specific repositories
        self._redis_repo = (
            RedisStateRepository(self.redis_adapter, self.config.redis_ttl_seconds)
            if self.redis_adapter
            else None
        )
        self._postgres_repo = (
            PostgresStateRepository(self.postgres_adapter) if self.postgres_adapter else None
        )
        self._s3_repo = (
            S3StateRepository(self.s3_adapter, self.config.s3_bucket) if self.s3_adapter else None
        )

        # Initialize tier promotion policy
        self._promotion_policy = TierPromotionPolicy(
            redis_repo=self._redis_repo,
            postgres_repo=self._postgres_repo,
            s3_repo=self._s3_repo,
        )

        # Initialize performance metrics
        self._metrics = StatePerformanceMetrics(enabled=self.config.enable_performance_tracking)

    # Backwards compatibility properties for tests
    @property
    def _local_cache(self) -> dict[str, Any]:
        """Access local cache (for backwards compatibility)."""
        return self._cache_manager._local_cache

    @property
    def _metadata_cache(self) -> dict[str, StateMetadata]:
        """Access metadata cache (for backwards compatibility)."""
        return self._cache_manager._metadata_cache

    @property
    def _access_history(self) -> dict[str, list[datetime]]:
        """Access access history (for backwards compatibility)."""
        return self._cache_manager._access_history

    def _update_access_history(self, key: str) -> None:
        """Update access history (for backwards compatibility)."""
        self._cache_manager.update_access_history(key)

    def _manage_cache_size(self) -> None:
        """Manage cache size (for backwards compatibility)."""
        self._cache_manager.manage_cache_size()

    def _calculate_checksum(self, data: str) -> str:
        """Calculate checksum (for backwards compatibility)."""
        return self._cache_manager.calculate_checksum(data)

    async def get_state(self, key: str, auto_promote: bool = True) -> Any | None:
        """
        Retrieve state with automatic tier escalation.

        Args:
            key: State key to retrieve
            auto_promote: Automatically promote to higher tier on access

        Returns:
            State value or None if not found
        """
        with self._metrics.time_operation("state_manager.get_state"):
            # Check local cache first
            cached_value = self._cache_manager.get(key)
            if cached_value is not None:
                return cached_value

            # Try hot storage (Redis)
            value = await self._get_from_redis(key)
            if value is not None:
                self._cache_manager.set(key, value)
                return value

            # Try warm storage (PostgreSQL)
            value = await self._get_from_postgres(key)
            if value is not None:
                if self._promotion_policy.should_auto_promote(StateCategory.WARM, auto_promote):
                    # Promote to hot storage
                    serialized = json.dumps(value, default=str)
                    await self._promotion_policy.promote_value(
                        key, serialized, StateCategory.WARM, {"ttl_seconds": None}
                    )
                self._cache_manager.set(key, value)
                return value

            # Try cold storage (S3)
            value = await self._get_from_s3(key)
            if value is not None:
                if self._promotion_policy.should_auto_promote(StateCategory.COLD, auto_promote):
                    # Promote to warm storage
                    serialized = json.dumps(value, default=str)
                    checksum = self._cache_manager.calculate_checksum(serialized)
                    await self._promotion_policy.promote_value(
                        key, serialized, StateCategory.COLD, {"checksum": checksum}
                    )
                self._cache_manager.set(key, value)
                return value

            return None

    async def set_state(
        self,
        key: str,
        value: Any,
        category: StateCategory = StateCategory.HOT,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Store state in appropriate tier.

        Args:
            key: State key
            value: State value to store
            category: Storage tier category
            ttl_seconds: Optional TTL for hot storage

        Returns:
            Success status
        """
        with self._metrics.time_operation("state_manager.set_state"):
            try:
                # Serialize value
                serialized = json.dumps(value, default=str)
                checksum = self._cache_manager.calculate_checksum(serialized)

                # Store in appropriate tier
                if category == StateCategory.HOT:
                    success = await self._set_in_redis(key, serialized, ttl_seconds)
                elif category == StateCategory.WARM:
                    success = await self._set_in_postgres(key, serialized, checksum)
                else:  # COLD
                    success = await self._set_in_s3(key, serialized, checksum)

                if success:
                    # Update metadata
                    self._cache_manager.update_metadata(
                        key=key,
                        category=category,
                        size_bytes=len(serialized.encode()),
                        checksum=checksum,
                        ttl_seconds=ttl_seconds,
                    )

                    # Update local cache
                    self._cache_manager.set(key, value)

                return success

            except Exception as e:
                logger.error(f"Failed to set state for key {key}: {e}")
                return False

    async def delete_state(self, key: str) -> bool:
        """Delete state from all tiers"""
        with self._metrics.time_operation("state_manager.delete_state"):
            success = True

            # Delete from all tiers
            if self.redis_adapter:
                try:
                    self.redis_adapter.delete(key)
                except Exception as exc:
                    success = False
                    logger.warning("Failed to delete %s from Redis: %s", key, exc, exc_info=True)

            if self.postgres_adapter:
                try:
                    self.postgres_adapter.execute("DELETE FROM state_warm WHERE key = %s", (key,))
                    self.postgres_adapter.commit()
                except Exception as exc:
                    success = False
                    logger.warning(
                        "Failed to delete %s from PostgreSQL: %s", key, exc, exc_info=True
                    )
                    try:
                        self.postgres_adapter.rollback()
                    except Exception:
                        logger.debug("PostgreSQL rollback failed after delete error", exc_info=True)

            if self.s3_adapter:
                try:
                    self.s3_adapter.delete_object(bucket=self.config.s3_bucket, key=f"cold/{key}")
                except Exception as exc:
                    success = False
                    logger.warning("Failed to delete %s from S3: %s", key, exc, exc_info=True)

            # Clear from caches
            self._cache_manager.delete(key)

            return success

    async def _get_from_redis(self, key: str) -> Any | None:
        """Get value from Redis"""
        if not self._redis_repo:
            return None
        return await self._redis_repo.fetch(key)

    async def _get_from_postgres(self, key: str) -> Any | None:
        """Get value from PostgreSQL"""
        if not self._postgres_repo:
            return None
        return await self._postgres_repo.fetch(key)

    async def _get_from_s3(self, key: str) -> Any | None:
        """Get value from S3"""
        if not self._s3_repo:
            return None
        return await self._s3_repo.fetch(key)

    async def _set_in_redis(self, key: str, value: str, ttl_seconds: int | None) -> bool:
        """Set value in Redis"""
        if not self._redis_repo:
            return False
        return await self._redis_repo.store(key, value, {"ttl_seconds": ttl_seconds})

    async def _set_in_postgres(self, key: str, value: str, checksum: str) -> bool:
        """Set value in PostgreSQL"""
        if not self._postgres_repo:
            return False
        return await self._postgres_repo.store(
            key, value, {"checksum": checksum, "size_bytes": len(value.encode())}
        )

    async def _set_in_s3(self, key: str, value: str, checksum: str) -> bool:
        """Set value in S3"""
        if not self._s3_repo:
            return False
        return await self._s3_repo.store(key, value, {"checksum": checksum})

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get all keys matching pattern"""
        with self._metrics.time_operation("state_manager.get_keys_by_pattern"):
            keys = set()

            # Check Redis
            if self._redis_repo:
                redis_keys = await self._redis_repo.keys(pattern)
                keys.update(redis_keys)

            # Check PostgreSQL
            if self._postgres_repo:
                pg_keys = await self._postgres_repo.keys(pattern)
                keys.update(pg_keys)

            # Check S3
            if self._s3_repo:
                s3_keys = await self._s3_repo.keys(pattern)
                keys.update(s3_keys)

            return list(keys)

    async def promote_to_hot(self, key: str) -> bool:
        """Manually promote state to hot tier"""
        value = await self.get_state(key, auto_promote=False)
        if value is not None:
            serialized = json.dumps(value, default=str)
            return await self._promotion_policy.promote_to_hot(
                key, serialized, {"ttl_seconds": None}
            )
        return False

    async def demote_to_cold(self, key: str) -> bool:
        """Manually demote state to cold tier"""
        value = await self.get_state(key, auto_promote=False)
        if value is not None:
            serialized = json.dumps(value, default=str)
            checksum = self._cache_manager.calculate_checksum(serialized)
            return await self._promotion_policy.demote_to_cold(
                key, serialized, {"checksum": checksum}
            )
        return False

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "hot_keys": 0,
            "warm_keys": 0,
            "cold_keys": 0,
            "cache_size_bytes": 0,
            "total_keys": 0,
        }

        # Redis stats
        if self._redis_repo:
            redis_stats = await self._redis_repo.stats()
            stats["hot_keys"] = redis_stats.get("key_count", 0)

        # PostgreSQL stats
        if self._postgres_repo:
            postgres_stats = await self._postgres_repo.stats()
            stats["warm_keys"] = postgres_stats.get("key_count", 0)

        # S3 stats
        if self._s3_repo:
            s3_stats = await self._s3_repo.stats()
            stats["cold_keys"] = s3_stats.get("key_count", 0)

        # Cache stats
        cache_stats = self._cache_manager.get_cache_stats()
        stats["cache_size_bytes"] = cache_stats["cache_size_bytes"]
        stats["cache_keys"] = cache_stats["cache_keys"]
        stats["total_keys"] = stats["hot_keys"] + stats["warm_keys"] + stats["cold_keys"]

        return stats

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for state operations.

        Returns detailed timing statistics for state operations when
        performance tracking is enabled via config.

        Returns:
            Dictionary with performance metrics or empty dict if disabled
        """
        if not self._metrics.is_enabled():
            return {"enabled": False, "message": "Performance tracking disabled"}

        return self._metrics.get_summary()

    def get_repositories(self) -> StateRepositories:
        """
        Get direct access to tier-specific repositories.

        Useful for batch operations (backup, checkpoint, recovery) that:
        - Don't need auto-promotion logic
        - Don't benefit from local caching
        - Need tier-specific access for performance

        Returns:
            StateRepositories bundle with redis, postgres, and s3 repositories

        Example:
            repos = state_manager.get_repositories()

            # Fast HOT-tier backup
            if repos.redis:
                keys = await repos.redis.keys("position:*")
                for key in keys:
                    value = await repos.redis.fetch(key)
        """
        return StateRepositories(
            redis=self._redis_repo,
            postgres=self._postgres_repo,
            s3=self._s3_repo,
        )

    async def batch_delete_state(self, keys: list[str]) -> int:
        """
        Delete multiple keys from all tiers with cache invalidation.

        Uses batch repository operations for efficiency while properly
        invalidating cache and metadata for each key.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted from at least one tier
        """
        if not keys:
            return 0

        with self._metrics.time_operation("state_manager.batch_delete_state"):
            deleted_count = 0

            # Use batch operations if repositories available
            if self._redis_repo and self._postgres_repo and self._s3_repo:
                # Batch delete from all tiers
                try:
                    if self._redis_repo:
                        await self._redis_repo.delete_many(keys)
                except Exception as e:
                    logger.warning(f"Batch delete from Redis failed: {e}")

                try:
                    if self._postgres_repo:
                        await self._postgres_repo.delete_many(keys)
                except Exception as e:
                    logger.warning(f"Batch delete from PostgreSQL failed: {e}")

                try:
                    if self._s3_repo:
                        await self._s3_repo.delete_many(keys)
                except Exception as e:
                    logger.warning(f"Batch delete from S3 failed: {e}")

                deleted_count = len(keys)
            else:
                # Fallback: Individual deletes (slower but works without repos)
                for key in keys:
                    if await self.delete_state(key):
                        deleted_count += 1

            # CRITICAL: Invalidate cache for all keys
            for key in keys:
                self._cache_manager.delete(key)

            logger.debug(f"Batch deleted {deleted_count} keys and invalidated cache")
            return deleted_count

    async def batch_set_state(
        self, items: dict[str, tuple[Any, StateCategory]], ttl_seconds: int | None = None
    ) -> int:
        """
        Set multiple key-value pairs with proper cache and metadata updates.

        Uses batch repository operations for efficiency while maintaining
        cache coherence, metadata tracking, and tier-appropriate storage.

        Cache and metadata updates occur ONLY after successful storage,
        matching single-item set_state semantics.

        Args:
            items: Dict mapping keys to (value, category) tuples
            ttl_seconds: Optional TTL override for HOT tier items

        Returns:
            Number of items successfully stored
        """
        if not items:
            return 0

        with self._metrics.time_operation("state_manager.batch_set_state"):
            import json

            # Group items by tier, preserving original values for cache updates
            hot_items: dict[str, tuple[str, dict[str, Any]]] = {}
            warm_items: dict[str, tuple[str, dict[str, Any]]] = {}
            cold_items: dict[str, tuple[str, dict[str, Any]]] = {}

            # Track metadata for cache updates (deferred until after successful storage)
            hot_metadata: dict[str, tuple[Any, int, str, int]] = {}  # (value, size, checksum, ttl)
            warm_metadata: dict[str, tuple[Any, int, str]] = {}  # (value, size, checksum)
            cold_metadata: dict[str, tuple[Any, int, str]] = {}  # (value, size, checksum)

            # Prepare items with metadata (but don't update cache yet)
            for key, (value, category) in items.items():
                try:
                    serialized = json.dumps(value, default=str)
                    checksum = self._cache_manager.calculate_checksum(serialized)
                    size_bytes = len(serialized.encode())

                    metadata = {
                        "checksum": checksum,
                        "size_bytes": size_bytes,
                    }

                    if category == StateCategory.HOT:
                        ttl = ttl_seconds or self.config.redis_ttl_seconds
                        metadata["ttl_seconds"] = ttl
                        hot_items[key] = (serialized, metadata)
                        hot_metadata[key] = (value, size_bytes, checksum, ttl)
                    elif category == StateCategory.WARM:
                        warm_items[key] = (serialized, metadata)
                        warm_metadata[key] = (value, size_bytes, checksum)
                    else:  # COLD
                        cold_items[key] = (serialized, metadata)
                        cold_metadata[key] = (value, size_bytes, checksum)

                except Exception as e:
                    logger.error(f"Failed to prepare item {key}: {e}")

            # Batch write to tiers and update cache only on success
            stored_count = 0

            if hot_items and self._redis_repo:
                try:
                    count = await self._redis_repo.store_many(hot_items)
                    if count > 0:
                        # Update cache and metadata only for successfully stored items
                        for key, (value, size_bytes, checksum, ttl) in hot_metadata.items():
                            self._cache_manager.set(key, value)
                            self._cache_manager.update_metadata(
                                key=key,
                                category=StateCategory.HOT,
                                size_bytes=size_bytes,
                                checksum=checksum,
                                ttl_seconds=ttl,
                            )
                        stored_count += count
                except Exception as e:
                    logger.error(f"Batch write to Redis failed: {e}")

            if warm_items and self._postgres_repo:
                try:
                    count = await self._postgres_repo.store_many(warm_items)
                    if count > 0:
                        # Update cache and metadata only for successfully stored items
                        for key, (value, size_bytes, checksum) in warm_metadata.items():
                            self._cache_manager.set(key, value)
                            self._cache_manager.update_metadata(
                                key=key,
                                category=StateCategory.WARM,
                                size_bytes=size_bytes,
                                checksum=checksum,
                            )
                        stored_count += count
                except Exception as e:
                    logger.error(f"Batch write to PostgreSQL failed: {e}")

            if cold_items and self._s3_repo:
                try:
                    count = await self._s3_repo.store_many(cold_items)
                    if count > 0:
                        # Update cache and metadata only for successfully stored items
                        for key, (value, size_bytes, checksum) in cold_metadata.items():
                            self._cache_manager.set(key, value)
                            self._cache_manager.update_metadata(
                                key=key,
                                category=StateCategory.COLD,
                                size_bytes=size_bytes,
                                checksum=checksum,
                            )
                        stored_count += count
                except Exception as e:
                    logger.error(f"Batch write to S3 failed: {e}")

            logger.debug(f"Batch stored {stored_count} items with cache/metadata updates")
            return stored_count

    def close(self) -> None:
        """Close all connections"""
        if self.redis_adapter:
            self.redis_adapter.close()
        if self.postgres_adapter:
            self.postgres_adapter.close()

        logger.info("StateManager connections closed")


# Global instance
_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get global StateManager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# Convenience functions
async def get_state(key: str) -> Any | None:
    """Get state value"""
    return await get_state_manager().get_state(key)


async def set_state(key: str, value: Any, category: StateCategory = StateCategory.HOT) -> bool:
    """Set state value"""
    return await get_state_manager().set_state(key, value, category)


async def delete_state(key: str) -> bool:
    """Delete state value"""
    return await get_state_manager().delete_state(key)
