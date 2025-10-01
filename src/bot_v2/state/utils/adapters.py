"""Storage adapters for StateManager.

Provides lightweight adapters for Redis, PostgreSQL, and S3 to enable
dependency injection and easier testing without module-level monkeypatching.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class RedisAdapter(ABC):
    """Abstract adapter for Redis operations."""

    @abstractmethod
    def ping(self) -> bool:
        """Ping Redis to verify connection."""
        pass

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Get value from Redis."""
        pass

    @abstractmethod
    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        """Set value in Redis with TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> int:
        """Delete key from Redis."""
        pass

    @abstractmethod
    def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    def dbsize(self) -> int:
        """Get number of keys in database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close Redis connection."""
        pass


class PostgresAdapter(ABC):
    """Abstract adapter for PostgreSQL operations."""

    @abstractmethod
    def execute(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute query and return results."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit transaction."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close PostgreSQL connection."""
        pass


class S3Adapter(ABC):
    """Abstract adapter for S3 operations."""

    @abstractmethod
    def head_bucket(self, bucket: str) -> dict[str, Any]:
        """Verify bucket exists."""
        pass

    @abstractmethod
    def get_object(self, bucket: str, key: str) -> dict[str, Any]:
        """Get object from S3."""
        pass

    @abstractmethod
    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        storage_class: str = "STANDARD",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Put object to S3."""
        pass

    @abstractmethod
    def delete_object(self, bucket: str, key: str) -> dict[str, Any]:
        """Delete object from S3."""
        pass

    @abstractmethod
    def list_objects_v2(self, bucket: str, prefix: str = "") -> dict[str, Any]:
        """List objects in S3 bucket."""
        pass


class DefaultRedisAdapter(RedisAdapter):
    """Default Redis adapter using redis-py."""

    def __init__(self, host: str, port: int, db: int):
        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
            )
        except ImportError as e:
            logger.warning("redis package not available")
            raise RuntimeError("Redis library not available") from e

    def ping(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.debug(f"Redis ping failed: {e}")
            return False

    def get(self, key: str) -> str | None:
        if not self._client:
            return None
        return self._client.get(key)

    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        if not self._client:
            return False
        try:
            self._client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            logger.error(f"Redis setex failed: {e}")
            return False

    def delete(self, key: str) -> int:
        if not self._client:
            return 0
        return self._client.delete(key)

    def keys(self, pattern: str) -> list[str]:
        if not self._client:
            return []
        return self._client.keys(pattern)

    def dbsize(self) -> int:
        if not self._client:
            return 0
        return self._client.dbsize()

    def close(self) -> None:
        if self._client:
            self._client.close()


class DefaultPostgresAdapter(PostgresAdapter):
    """Default PostgreSQL adapter using psycopg2."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self._conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                cursor_factory=RealDictCursor,
            )
        except ImportError as e:
            logger.warning("psycopg2 package not available")
            raise RuntimeError("PostgreSQL library not available") from e

    def execute(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        if not self._conn:
            return []

        with self._conn.cursor() as cursor:
            cursor.execute(query, params)
            # Only fetch if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            return []

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn:
            self._conn.rollback()

    def close(self) -> None:
        if self._conn:
            self._conn.close()


class DefaultS3Adapter(S3Adapter):
    """Default S3 adapter using boto3."""

    def __init__(self, region: str):
        try:
            import boto3

            self._client = boto3.client("s3", region_name=region)
        except ImportError as e:
            logger.warning("boto3 package not available")
            raise RuntimeError("S3 library not available") from e

    def head_bucket(self, bucket: str) -> dict[str, Any]:
        if not self._client:
            raise Exception("S3 client not available")
        return self._client.head_bucket(Bucket=bucket)

    def get_object(self, bucket: str, key: str) -> dict[str, Any]:
        if not self._client:
            raise Exception("S3 client not available")
        return self._client.get_object(Bucket=bucket, Key=key)

    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        storage_class: str = "STANDARD",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self._client:
            raise Exception("S3 client not available")

        kwargs = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "StorageClass": storage_class,
        }
        if metadata:
            kwargs["Metadata"] = metadata

        return self._client.put_object(**kwargs)

    def delete_object(self, bucket: str, key: str) -> dict[str, Any]:
        if not self._client:
            raise Exception("S3 client not available")
        return self._client.delete_object(Bucket=bucket, Key=key)

    def list_objects_v2(self, bucket: str, prefix: str = "") -> dict[str, Any]:
        if not self._client:
            raise Exception("S3 client not available")
        return self._client.list_objects_v2(Bucket=bucket, Prefix=prefix)
