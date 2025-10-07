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

    # Batch operations
    @abstractmethod
    def mget(self, keys: list[str]) -> list[str | None]:
        """Get multiple values from Redis."""
        pass

    @abstractmethod
    def mset(self, mapping: dict[str, str]) -> bool:
        """Set multiple key-value pairs in Redis."""
        pass

    @abstractmethod
    def msetex(self, mapping: dict[str, str], ttl_seconds: int) -> bool:
        """Set multiple key-value pairs with TTL using pipeline."""
        pass

    @abstractmethod
    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys from Redis."""
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

    # Batch operations
    @abstractmethod
    def executemany(self, query: str, params_list: list[tuple]) -> None:
        """Execute query with multiple parameter sets."""
        pass

    @abstractmethod
    def batch_upsert(self, table: str, key_column: str, records: list[dict[str, Any]]) -> int:
        """Batch upsert records into table."""
        pass

    @abstractmethod
    def batch_delete(self, table: str, key_column: str, keys: list[str]) -> int:
        """Delete multiple records by key."""
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

    # Batch operations
    @abstractmethod
    def delete_objects(self, bucket: str, keys: list[str]) -> dict[str, Any]:
        """Delete multiple objects from S3 (up to 1000 keys)."""
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

    # Batch operations
    def mget(self, keys: list[str]) -> list[str | None]:
        if not self._client or not keys:
            return []
        try:
            return self._client.mget(keys)
        except Exception as e:
            logger.error(f"Redis mget failed: {e}")
            return []

    def mset(self, mapping: dict[str, str]) -> bool:
        if not self._client or not mapping:
            return False
        try:
            return self._client.mset(mapping)
        except Exception as e:
            logger.error(f"Redis mset failed: {e}")
            return False

    def msetex(self, mapping: dict[str, str], ttl_seconds: int) -> bool:
        """Set multiple key-value pairs with TTL using pipeline."""
        if not self._client or not mapping:
            return False
        try:
            pipe = self._client.pipeline()
            for key, value in mapping.items():
                pipe.setex(key, ttl_seconds, value)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis msetex failed: {e}")
            return False

    def delete_many(self, keys: list[str]) -> int:
        if not self._client or not keys:
            return 0
        try:
            return self._client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete_many failed: {e}")
            return 0


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

    # Batch operations
    def executemany(self, query: str, params_list: list[tuple]) -> None:
        """Execute query with multiple parameter sets."""
        if not self._conn or not params_list:
            return

        try:
            with self._conn.cursor() as cursor:
                cursor.executemany(query, params_list)
        except Exception as e:
            logger.error(f"PostgreSQL executemany failed: {e}")
            raise

    def batch_upsert(self, table: str, key_column: str, records: list[dict[str, Any]]) -> int:
        """Batch upsert records into table."""
        if not self._conn or not records:
            return 0

        try:
            # Build column list from first record
            columns = list(records[0].keys())
            col_names = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))

            # Build update clause excluding key column
            update_parts = [f"{col} = EXCLUDED.{col}" for col in columns if col != key_column]

            # Add critical fields for tier promotion and versioning
            # (matches single-item store() behavior in repositories.py:293-294)
            update_parts.append("last_accessed = CURRENT_TIMESTAMP")
            update_parts.append(f"version = {table}.version + 1")

            update_clause = ", ".join(update_parts)

            query = f"""
                INSERT INTO {table} ({col_names})
                VALUES ({placeholders})
                ON CONFLICT ({key_column}) DO UPDATE SET
                    {update_clause}
            """

            params_list = [tuple(rec[col] for col in columns) for rec in records]

            with self._conn.cursor() as cursor:
                cursor.executemany(query, params_list)

            return len(records)
        except Exception as e:
            logger.error(f"PostgreSQL batch_upsert failed: {e}")
            raise

    def batch_delete(self, table: str, key_column: str, keys: list[str]) -> int:
        """Delete multiple records by key."""
        if not self._conn or not keys:
            return 0

        try:
            placeholders = ", ".join(["%s"] * len(keys))
            query = f"DELETE FROM {table} WHERE {key_column} IN ({placeholders})"

            with self._conn.cursor() as cursor:
                cursor.execute(query, tuple(keys))

            return len(keys)
        except Exception as e:
            logger.error(f"PostgreSQL batch_delete failed: {e}")
            raise


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

    # Batch operations
    def delete_objects(self, bucket: str, keys: list[str]) -> dict[str, Any]:
        """Delete multiple objects from S3 (up to 1000 keys per request)."""
        if not self._client:
            raise Exception("S3 client not available")
        if not keys:
            return {"Deleted": [], "Errors": []}

        # S3 delete_objects accepts max 1000 keys per request
        if len(keys) > 1000:
            logger.warning(
                f"delete_objects received {len(keys)} keys, only first 1000 will be deleted"
            )
            keys = keys[:1000]

        delete_request = {"Objects": [{"Key": key} for key in keys]}

        try:
            response = self._client.delete_objects(Bucket=bucket, Delete=delete_request)
            return response
        except Exception as e:
            logger.error(f"S3 delete_objects failed: {e}")
            raise
