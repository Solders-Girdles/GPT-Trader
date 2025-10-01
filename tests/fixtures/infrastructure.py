"""Shared infrastructure fixtures for state management and storage.

Provides mock adapters and configurations for testing state storage across
multiple tiers (Redis, PostgreSQL, S3). These fixtures reduce test overhead
while maintaining narrative clarity about system behavior.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.state.state_manager import StateConfig
from bot_v2.state.utils.adapters import PostgresAdapter, RedisAdapter, S3Adapter


@pytest.fixture
def state_config() -> StateConfig:
    """Create test state configuration.

    Provides a standard configuration for testing tiered storage systems.
    Enables compression but disables encryption for test performance.
    """
    return StateConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_ttl_seconds=3600,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_database="test_db",
        postgres_user="test_user",
        postgres_password="test_pass",
        s3_bucket="test-bucket",
        s3_region="us-east-1",
        enable_compression=True,
        enable_encryption=False,
        cache_size_mb=100,
    )


@pytest.fixture
def mock_redis_adapter() -> Mock:
    """Create mock Redis adapter.

    Simulates HOT tier storage with sub-second access times.
    All operations succeed by default - configure failures per test.
    """
    adapter = Mock(spec=RedisAdapter)
    adapter.ping.return_value = True
    adapter.get.return_value = None
    adapter.setex.return_value = True
    adapter.delete.return_value = 1
    adapter.keys.return_value = []
    adapter.dbsize.return_value = 0
    adapter.close.return_value = None
    return adapter


@pytest.fixture
def mock_postgres_adapter() -> Mock:
    """Create mock Postgres adapter.

    Simulates WARM tier storage for recent data access.
    All operations succeed by default - configure failures per test.
    """
    adapter = Mock(spec=PostgresAdapter)
    adapter.execute.return_value = []
    adapter.commit.return_value = None
    adapter.rollback.return_value = None
    adapter.close.return_value = None
    return adapter


@pytest.fixture
def mock_s3_adapter() -> Mock:
    """Create mock S3 adapter.

    Simulates COLD tier storage for archival data.
    All operations succeed by default - configure failures per test.
    """
    adapter = Mock(spec=S3Adapter)
    adapter.head_bucket.return_value = {}
    adapter.get_object.return_value = {"Body": Mock()}
    adapter.put_object.return_value = {}
    adapter.delete_object.return_value = {}
    adapter.list_objects_v2.return_value = {}
    return adapter


# Failing adapter implementations for deterministic degradation tests


class FailingRedisAdapter(RedisAdapter):
    """Redis adapter that fails deterministically for degradation testing."""

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        return False

    def delete(self, key: str) -> int:
        raise RuntimeError("redis offline")

    def keys(self, pattern: str) -> list[str]:
        raise RuntimeError("redis pattern failure")

    def dbsize(self) -> int:
        raise RuntimeError("redis stats failure")

    def close(self) -> None:
        pass


class FailingPostgresAdapter(PostgresAdapter):
    """Postgres adapter that fails deterministically for degradation testing."""

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        raise RuntimeError("pg execute failure")

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        raise RuntimeError("pg rollback failure")

    def close(self) -> None:
        pass


class FailingS3Adapter(S3Adapter):
    """S3 adapter that fails deterministically for degradation testing."""

    def head_bucket(self, bucket: str) -> dict:
        raise RuntimeError("s3 head failure")

    def get_object(self, bucket: str, key: str) -> dict:
        raise RuntimeError("s3 get failure")

    def put_object(
        self, bucket: str, key: str, body: bytes, storage_class: str = "STANDARD", metadata: dict | None = None
    ) -> dict:
        raise RuntimeError("s3 put failure")

    def delete_object(self, bucket: str, key: str) -> dict:
        raise RuntimeError("s3 delete failure")

    def list_objects_v2(self, bucket: str, prefix: str = "") -> dict:
        raise RuntimeError("s3 list failure")


@pytest.fixture
def failing_redis_adapter() -> FailingRedisAdapter:
    """Create a Redis adapter that fails deterministically.

    Used for testing graceful degradation when Redis is unavailable.
    """
    return FailingRedisAdapter()


@pytest.fixture
def failing_postgres_adapter() -> FailingPostgresAdapter:
    """Create a Postgres adapter that fails deterministically.

    Used for testing graceful degradation when PostgreSQL is unavailable.
    """
    return FailingPostgresAdapter()


@pytest.fixture
def failing_s3_adapter() -> FailingS3Adapter:
    """Create an S3 adapter that fails deterministically.

    Used for testing graceful degradation when S3 is unavailable.
    """
    return FailingS3Adapter()
