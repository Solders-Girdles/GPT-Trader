"""
State Manager for Bot V2 Trading System

Provides multi-tier state management with hot (Redis), warm (PostgreSQL),
and cold (S3) storage layers for optimal performance and cost efficiency.
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# External dependencies (installed via requirements)
try:
    import boto3
    import psycopg2
    import redis
    from psycopg2.extras import RealDictCursor
except ImportError as e:
    logging.warning(f"Optional dependency not installed: {e}")
    redis = None
    psycopg2 = None
    boto3 = None

logger = logging.getLogger(__name__)


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


class StateManager:
    """
    Central state coordination with multi-tier storage.
    Automatically promotes/demotes data between tiers based on access patterns.
    """

    def __init__(self, config: StateConfig | None = None) -> None:
        self.config = config or StateConfig()
        self._lock = threading.Lock()
        self._local_cache: dict[str, Any] = {}
        self._access_history: dict[str, list[datetime]] = {}
        self._metadata_cache: dict[str, StateMetadata] = {}

        # Initialize storage backends
        self._init_redis()
        self._init_postgres()
        self._init_s3()

    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True,
                )
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.redis_client = None
        else:
            self.redis_client = None
            logger.warning("Redis not available")

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection"""
        if psycopg2:
            try:
                self.pg_conn = psycopg2.connect(
                    host=self.config.postgres_host,
                    port=self.config.postgres_port,
                    database=self.config.postgres_database,
                    user=self.config.postgres_user,
                    password=self.config.postgres_password,
                    cursor_factory=RealDictCursor,
                )
                # Create tables if not exist
                self._create_postgres_tables()
                logger.info("PostgreSQL connection established")
            except Exception as e:
                logger.warning(f"PostgreSQL initialization failed: {e}")
                self.pg_conn = None
        else:
            self.pg_conn = None
            logger.warning("PostgreSQL not available")

    def _init_s3(self) -> None:
        """Initialize S3 client"""
        if boto3:
            try:
                self.s3_client = boto3.client("s3", region_name=self.config.s3_region)
                # Verify bucket exists
                self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                logger.info("S3 connection established")
            except Exception as e:
                logger.warning(f"S3 initialization failed: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
            logger.warning("S3 not available")

    def _create_postgres_tables(self) -> None:
        """Create necessary PostgreSQL tables"""
        if not self.pg_conn:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS state_warm (
            key VARCHAR(255) PRIMARY KEY,
            data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            size_bytes INTEGER,
            checksum VARCHAR(64),
            version INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_state_warm_last_accessed
        ON state_warm(last_accessed);

        CREATE TABLE IF NOT EXISTS state_metadata (
            key VARCHAR(255) PRIMARY KEY,
            category VARCHAR(10),
            location VARCHAR(255),
            created_at TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            size_bytes INTEGER,
            checksum VARCHAR(64)
        );
        """

        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                self.pg_conn.commit()
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
            self.pg_conn.rollback()

    async def get_state(self, key: str, auto_promote: bool = True) -> Any | None:
        """
        Retrieve state with automatic tier escalation.

        Args:
            key: State key to retrieve
            auto_promote: Automatically promote to higher tier on access

        Returns:
            State value or None if not found
        """
        # Check local cache first
        if key in self._local_cache:
            self._update_access_history(key)
            return self._local_cache[key]

        # Try hot storage (Redis)
        value = await self._get_from_redis(key)
        if value is not None:
            self._update_access_history(key)
            self._local_cache[key] = value
            return value

        # Try warm storage (PostgreSQL)
        value = await self._get_from_postgres(key)
        if value is not None:
            self._update_access_history(key)
            if auto_promote:
                # Promote to hot storage
                await self.set_state(key, value, StateCategory.HOT)
            self._local_cache[key] = value
            return value

        # Try cold storage (S3)
        value = await self._get_from_s3(key)
        if value is not None:
            self._update_access_history(key)
            if auto_promote:
                # Promote to warm storage
                await self.set_state(key, value, StateCategory.WARM)
            self._local_cache[key] = value
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
        try:
            # Serialize value
            serialized = json.dumps(value, default=str)
            checksum = self._calculate_checksum(serialized)

            # Store in appropriate tier
            if category == StateCategory.HOT:
                success = await self._set_in_redis(key, serialized, ttl_seconds)
            elif category == StateCategory.WARM:
                success = await self._set_in_postgres(key, serialized, checksum)
            else:  # COLD
                success = await self._set_in_s3(key, serialized, checksum)

            if success:
                # Update metadata
                metadata = StateMetadata(
                    key=key,
                    category=category,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    size_bytes=len(serialized.encode()),
                    checksum=checksum,
                    version=1,
                    ttl_seconds=ttl_seconds,
                )
                self._metadata_cache[key] = metadata

                # Update local cache
                self._local_cache[key] = value

                # Manage cache size
                self._manage_cache_size()

            return success

        except Exception as e:
            logger.error(f"Failed to set state for key {key}: {e}")
            return False

    async def delete_state(self, key: str) -> bool:
        """Delete state from all tiers"""
        success = True

        # Delete from all tiers
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as exc:
                success = False
                logger.warning("Failed to delete %s from Redis: %s", key, exc, exc_info=True)

        if self.pg_conn:
            try:
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("DELETE FROM state_warm WHERE key = %s", (key,))
                    self.pg_conn.commit()
            except Exception as exc:
                success = False
                logger.warning("Failed to delete %s from PostgreSQL: %s", key, exc, exc_info=True)
                try:
                    self.pg_conn.rollback()
                except Exception:
                    logger.debug("PostgreSQL rollback failed after delete error", exc_info=True)

        if self.s3_client:
            try:
                self.s3_client.delete_object(Bucket=self.config.s3_bucket, Key=f"cold/{key}")
            except Exception as exc:
                success = False
                logger.warning("Failed to delete %s from S3: %s", key, exc, exc_info=True)

        # Clear from caches
        self._local_cache.pop(key, None)
        self._metadata_cache.pop(key, None)
        self._access_history.pop(key, None)

        return success

    async def _get_from_redis(self, key: str) -> Any | None:
        """Get value from Redis"""
        if not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Redis get failed for {key}: {e}")

        return None

    async def _get_from_postgres(self, key: str) -> Any | None:
        """Get value from PostgreSQL"""
        if not self.pg_conn:
            return None

        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute("SELECT data FROM state_warm WHERE key = %s", (key,))
                result = cursor.fetchone()
                if result:
                    # Update last accessed time
                    cursor.execute(
                        "UPDATE state_warm SET last_accessed = %s WHERE key = %s",
                        (datetime.utcnow(), key),
                    )
                    self.pg_conn.commit()
                    return result["data"]
        except Exception as e:
            logger.debug(f"PostgreSQL get failed for {key}: {e}")
            self.pg_conn.rollback()

        return None

    async def _get_from_s3(self, key: str) -> Any | None:
        """Get value from S3"""
        if not self.s3_client:
            return None

        try:
            response = self.s3_client.get_object(Bucket=self.config.s3_bucket, Key=f"cold/{key}")
            data = response["Body"].read().decode("utf-8")
            return json.loads(data)
        except Exception as e:
            logger.debug(f"S3 get failed for {key}: {e}")

        return None

    async def _set_in_redis(self, key: str, value: str, ttl_seconds: int | None) -> bool:
        """Set value in Redis"""
        if not self.redis_client:
            return False

        try:
            ttl = ttl_seconds or self.config.redis_ttl_seconds
            self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis set failed for {key}: {e}")
            return False

    async def _set_in_postgres(self, key: str, value: str, checksum: str) -> bool:
        """Set value in PostgreSQL"""
        if not self.pg_conn:
            return False

        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute(
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
                    (key, value, checksum, len(value.encode())),
                )
                self.pg_conn.commit()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL set failed for {key}: {e}")
            self.pg_conn.rollback()
            return False

    async def _set_in_s3(self, key: str, value: str, checksum: str) -> bool:
        """Set value in S3"""
        if not self.s3_client:
            return False

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=f"cold/{key}",
                Body=value.encode(),
                StorageClass="STANDARD_IA",
                Metadata={"checksum": checksum},
            )
            return True
        except Exception as e:
            logger.error(f"S3 set failed for {key}: {e}")
            return False

    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA256 checksum"""
        return hashlib.sha256(data.encode()).hexdigest()

    def _update_access_history(self, key: str) -> None:
        """Update access history for tier management"""
        if key not in self._access_history:
            self._access_history[key] = []

        self._access_history[key].append(datetime.utcnow())

        # Keep only last 100 accesses
        if len(self._access_history[key]) > 100:
            self._access_history[key] = self._access_history[key][-100:]

    def _manage_cache_size(self) -> None:
        """Manage local cache size"""
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

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get all keys matching pattern"""
        keys = set()

        # Check Redis
        if self.redis_client:
            try:
                redis_keys = self.redis_client.keys(pattern)
                keys.update(redis_keys)
            except Exception as exc:
                logger.debug(
                    "Redis key lookup failed for pattern %s: %s", pattern, exc, exc_info=True
                )

        # Check PostgreSQL
        if self.pg_conn:
            try:
                sql_pattern = pattern.replace("*", "%")
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT key FROM state_warm WHERE key LIKE %s", (sql_pattern,))
                    pg_keys = [row["key"] for row in cursor.fetchall()]
                    keys.update(pg_keys)
            except Exception as exc:
                logger.debug("PostgreSQL key lookup failed for %s: %s", pattern, exc, exc_info=True)

        # Check S3 (limited pattern matching)
        if self.s3_client:
            try:
                prefix = pattern.split("*")[0] if "*" in pattern else pattern
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config.s3_bucket, Prefix=f"cold/{prefix}"
                )
                if "Contents" in response:
                    s3_keys = [obj["Key"].replace("cold/", "") for obj in response["Contents"]]
                    keys.update(s3_keys)
            except Exception as exc:
                logger.debug("S3 key lookup failed for %s: %s", pattern, exc, exc_info=True)

        return list(keys)

    async def promote_to_hot(self, key: str) -> bool:
        """Manually promote state to hot tier"""
        value = await self.get_state(key, auto_promote=False)
        if value is not None:
            return await self.set_state(key, value, StateCategory.HOT)
        return False

    async def demote_to_cold(self, key: str) -> bool:
        """Manually demote state to cold tier"""
        value = await self.get_state(key, auto_promote=False)
        if value is not None:
            # Delete from hot/warm tiers
            if self.redis_client:
                self.redis_client.delete(key)
            if self.pg_conn:
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("DELETE FROM state_warm WHERE key = %s", (key,))
                    self.pg_conn.commit()

            # Store in cold tier
            return await self.set_state(key, value, StateCategory.COLD)
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
        if self.redis_client:
            try:
                stats["hot_keys"] = self.redis_client.dbsize()
            except Exception as exc:
                logger.debug("Redis stats collection failed: %s", exc, exc_info=True)

        # PostgreSQL stats
        if self.pg_conn:
            try:
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) as count FROM state_warm")
                    result = cursor.fetchone()
                    stats["warm_keys"] = result["count"]
            except Exception as exc:
                logger.debug("PostgreSQL stats collection failed: %s", exc, exc_info=True)

        # S3 stats
        if self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config.s3_bucket, Prefix="cold/"
                )
                stats["cold_keys"] = response.get("KeyCount", 0)
            except Exception as exc:
                logger.debug("S3 stats collection failed: %s", exc, exc_info=True)

        # Cache stats
        stats["cache_size_bytes"] = sum(
            len(json.dumps(v, default=str).encode()) for v in self._local_cache.values()
        )
        stats["cache_keys"] = len(self._local_cache)
        stats["total_keys"] = stats["hot_keys"] + stats["warm_keys"] + stats["cold_keys"]

        return stats

    def close(self) -> None:
        """Close all connections"""
        if self.redis_client:
            self.redis_client.close()
        if self.pg_conn:
            self.pg_conn.close()

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
