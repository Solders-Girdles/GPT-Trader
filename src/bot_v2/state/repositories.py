"""
State Repositories for Tier-Specific Storage

Encapsulates tier-specific storage operations with a shared interface
for Redis (HOT), PostgreSQL (WARM), and S3 (COLD) tiers.
"""

import json
import logging
from datetime import datetime
from typing import Any, Protocol

from bot_v2.state.utils.adapters import PostgresAdapter, RedisAdapter, S3Adapter

logger = logging.getLogger(__name__)


class StateRepository(Protocol):
    """
    Protocol defining the interface for tier-specific state repositories.

    All repositories must implement these methods for consistent access.
    """

    async def fetch(self, key: str) -> Any | None:
        """Fetch state value by key."""
        ...

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """Store state value with metadata."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete state by key."""
        ...

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        ...

    async def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        ...

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items at once.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        ...

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys at once.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        ...


class RedisStateRepository:
    """
    Redis repository for HOT tier state storage.

    Provides sub-second access times for frequently accessed data.
    """

    def __init__(self, adapter: RedisAdapter, default_ttl: int = 3600) -> None:
        """
        Initialize Redis repository.

        Args:
            adapter: Redis adapter instance
            default_ttl: Default TTL for Redis keys in seconds
        """
        self.adapter = adapter
        self.default_ttl = default_ttl

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from Redis.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        try:
            value = self.adapter.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Redis fetch failed for {key}: {e}")

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
        try:
            ttl = metadata.get("ttl_seconds", self.default_ttl)
            return self.adapter.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Redis store failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from Redis.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.adapter.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")
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

        try:
            return self.adapter.delete_many(keys)
        except Exception as e:
            logger.error(f"Redis delete_many failed: {e}")
            return 0


class PostgresStateRepository:
    """
    PostgreSQL repository for WARM tier state storage.

    Provides recent data access with ~5s latency.
    """

    def __init__(self, adapter: PostgresAdapter) -> None:
        """
        Initialize PostgreSQL repository.

        Args:
            adapter: PostgreSQL adapter instance
        """
        self.adapter = adapter

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from PostgreSQL.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        try:
            results = self.adapter.execute("SELECT data FROM state_warm WHERE key = %s", (key,))
            if results:
                result = results[0]
                # Update last accessed time
                self.adapter.execute(
                    "UPDATE state_warm SET last_accessed = %s WHERE key = %s",
                    (datetime.utcnow(), key),
                )
                self.adapter.commit()
                return result["data"]
        except Exception as e:
            logger.debug(f"PostgreSQL fetch failed for {key}: {e}")
            self.adapter.rollback()

        return None

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Store state in PostgreSQL.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata dict containing 'checksum' and 'size_bytes'

        Returns:
            True if successful, False otherwise
        """
        try:
            checksum = metadata.get("checksum", "")
            size_bytes = metadata.get("size_bytes", len(value.encode()))

            self.adapter.execute(
                """
                INSERT INTO state_warm (key, data, checksum, size_bytes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    data = EXCLUDED.data,
                    checksum = EXCLUDED.checksum,
                    size_bytes = EXCLUDED.size_bytes,
                    last_accessed = CURRENT_TIMESTAMP,
                    version = state_warm.version + 1
            """,
                (key, value, checksum, size_bytes),
            )
            self.adapter.commit()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL store failed for {key}: {e}")
            self.adapter.rollback()
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from PostgreSQL.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.adapter.execute("DELETE FROM state_warm WHERE key = %s", (key,))
            self.adapter.commit()
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL delete failed for {key}: {e}")
            try:
                self.adapter.rollback()
            except Exception:
                logger.debug("PostgreSQL rollback failed after delete error", exc_info=True)
            return False

    async def keys(self, pattern: str) -> list[str]:
        """
        Get PostgreSQL keys matching pattern.

        Args:
            pattern: Key pattern (converts * to SQL %)

        Returns:
            List of matching keys
        """
        try:
            sql_pattern = pattern.replace("*", "%")
            results = self.adapter.execute(
                "SELECT key FROM state_warm WHERE key LIKE %s", (sql_pattern,)
            )
            return [row["key"] for row in results]
        except Exception as e:
            logger.debug(f"PostgreSQL key lookup failed for {pattern}: {e}")
            return []

    async def stats(self) -> dict[str, Any]:
        """
        Get PostgreSQL storage statistics.

        Returns:
            Dict containing 'key_count' and other stats
        """
        try:
            results = self.adapter.execute("SELECT COUNT(*) as count FROM state_warm")
            if results:
                result = results[0]
                return {"key_count": result["count"]}
        except Exception as e:
            logger.debug(f"PostgreSQL stats collection failed: {e}")

        return {"key_count": 0}

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items in PostgreSQL using batch upsert.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        if not items:
            return set()

        try:
            records = []
            for key, (value, metadata) in items.items():
                checksum = metadata.get("checksum", "")
                size_bytes = metadata.get("size_bytes", len(value.encode()))
                records.append(
                    {
                        "key": key,
                        "data": value,
                        "checksum": checksum,
                        "size_bytes": size_bytes,
                    }
                )

            count = self.adapter.batch_upsert("state_warm", "key", records)
            self.adapter.commit()
            # All items succeeded (PostgreSQL batch_upsert is transactional)
            return set(items.keys()) if count > 0 else set()
        except Exception as e:
            logger.error(f"PostgreSQL store_many failed: {e}")
            self.adapter.rollback()
            return set()

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from PostgreSQL.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        if not keys:
            return 0

        try:
            count = self.adapter.batch_delete("state_warm", "key", keys)
            self.adapter.commit()
            return count
        except Exception as e:
            logger.error(f"PostgreSQL delete_many failed: {e}")
            try:
                self.adapter.rollback()
            except Exception:
                logger.debug("PostgreSQL rollback failed after delete_many error", exc_info=True)
            return 0


class S3StateRepository:
    """
    S3 repository for COLD tier state storage.

    Provides long-term archival storage with lower access times.
    """

    def __init__(self, adapter: S3Adapter, bucket: str, prefix: str = "cold/") -> None:
        """
        Initialize S3 repository.

        Args:
            adapter: S3 adapter instance
            bucket: S3 bucket name
            prefix: Key prefix for cold storage objects
        """
        self.adapter = adapter
        self.bucket = bucket
        self.prefix = prefix

    def _build_key(self, key: str) -> str:
        """Build full S3 key with prefix."""
        return f"{self.prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Strip prefix from S3 key."""
        return key.replace(self.prefix, "")

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from S3.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        try:
            response = self.adapter.get_object(bucket=self.bucket, key=self._build_key(key))
            data = response["Body"].read().decode("utf-8")
            return json.loads(data)
        except Exception as e:
            logger.debug(f"S3 fetch failed for {key}: {e}")

        return None

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Store state in S3.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata dict containing 'checksum'

        Returns:
            True if successful, False otherwise
        """
        try:
            checksum = metadata.get("checksum", "")
            self.adapter.put_object(
                bucket=self.bucket,
                key=self._build_key(key),
                body=value.encode(),
                storage_class="STANDARD_IA",
                metadata={"checksum": checksum},
            )
            return True
        except Exception as e:
            logger.error(f"S3 store failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from S3.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.adapter.delete_object(bucket=self.bucket, key=self._build_key(key))
            return True
        except Exception as e:
            logger.warning(f"S3 delete failed for {key}: {e}")
            return False

    async def keys(self, pattern: str) -> list[str]:
        """
        Get S3 keys matching pattern.

        Args:
            pattern: Key pattern (limited support, uses prefix)

        Returns:
            List of matching keys
        """
        try:
            prefix = pattern.split("*")[0] if "*" in pattern else pattern
            response = self.adapter.list_objects_v2(
                bucket=self.bucket, prefix=self._build_key(prefix)
            )
            if "Contents" in response:
                return [self._strip_prefix(obj["Key"]) for obj in response["Contents"]]
        except Exception as e:
            logger.debug(f"S3 key lookup failed for {pattern}: {e}")

        return []

    async def stats(self) -> dict[str, Any]:
        """
        Get S3 storage statistics.

        Returns:
            Dict containing 'key_count' and other stats
        """
        try:
            response = self.adapter.list_objects_v2(bucket=self.bucket, prefix=self.prefix)
            return {"key_count": response.get("KeyCount", 0)}
        except Exception as e:
            logger.debug(f"S3 stats collection failed: {e}")

        return {"key_count": 0}

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items in S3.

        Note: S3 doesn't have batch put, so this iterates sequentially.
        For true parallelism, consider using concurrent uploads at higher level.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        if not items:
            return set()

        successful_keys = set()
        for key, (value, metadata) in items.items():
            try:
                checksum = metadata.get("checksum", "")
                self.adapter.put_object(
                    bucket=self.bucket,
                    key=self._build_key(key),
                    body=value.encode(),
                    storage_class="STANDARD_IA",
                    metadata={"checksum": checksum},
                )
                successful_keys.add(key)
            except Exception as e:
                logger.error(f"S3 store failed for {key}: {e}")

        return successful_keys

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from S3 using batch delete.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        if not keys:
            return 0

        try:
            # Build full S3 keys with prefix
            full_keys = [self._build_key(key) for key in keys]

            # S3 batch delete handles up to 1000 keys
            response = self.adapter.delete_objects(bucket=self.bucket, keys=full_keys)

            # Count successful deletions
            deleted = len(response.get("Deleted", []))
            errors = response.get("Errors", [])

            if errors:
                logger.warning(f"S3 delete_many had {len(errors)} errors")

            return deleted
        except Exception as e:
            logger.error(f"S3 delete_many failed: {e}")
            return 0
