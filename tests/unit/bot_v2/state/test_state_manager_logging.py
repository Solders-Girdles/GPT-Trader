"""Regression tests for StateManager logging pathways."""

from __future__ import annotations

import pytest

from bot_v2.state import state_manager as sm
from tests.fixtures.infrastructure import (
    FailingPostgresAdapter,
    FailingRedisAdapter,
    FailingS3Adapter,
)


@pytest.fixture
def manager() -> sm.StateManager:
    """Provide a StateManager with failing backends for logging tests."""
    return sm.StateManager(
        config=sm.StateConfig(),
        redis_adapter=FailingRedisAdapter(),
        postgres_adapter=FailingPostgresAdapter(),
        s3_adapter=FailingS3Adapter(),
    )


@pytest.mark.asyncio
async def test_delete_state_logs_backend_failures(manager: sm.StateManager, caplog):
    from bot_v2.state.repositories import (
        RedisStateRepository,
        PostgresStateRepository,
        S3StateRepository,
    )

    # Manager already has failing adapters from fixture
    # Note: Postgres/S3 may be None after initialization validation, so re-assign
    manager.redis_adapter = FailingRedisAdapter()
    manager.postgres_adapter = FailingPostgresAdapter()
    manager.s3_adapter = FailingS3Adapter()

    # Recreate repositories with failing adapters
    manager._redis_repo = RedisStateRepository(manager.redis_adapter, 3600)
    manager._postgres_repo = PostgresStateRepository(manager.postgres_adapter)
    manager._s3_repo = S3StateRepository(manager.s3_adapter, "test-bucket")

    with caplog.at_level("DEBUG"):
        success = await manager.delete_state("portfolio:test")

    assert success is False
    assert any("Failed to delete portfolio:test from Redis" in msg for msg in caplog.messages)
    assert any("Failed to delete portfolio:test from PostgreSQL" in msg for msg in caplog.messages)
    assert any("Failed to delete portfolio:test from S3" in msg for msg in caplog.messages)
    assert any("PostgreSQL rollback failed after delete error" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_get_keys_by_pattern_handles_backend_errors(manager: sm.StateManager, caplog):
    from bot_v2.state.repositories import (
        RedisStateRepository,
        PostgresStateRepository,
        S3StateRepository,
    )

    # Re-assign to ensure all adapters are present for this test
    manager.redis_adapter = FailingRedisAdapter()
    manager.postgres_adapter = FailingPostgresAdapter()
    manager.s3_adapter = FailingS3Adapter()

    # Recreate repositories with failing adapters
    manager._redis_repo = RedisStateRepository(manager.redis_adapter, 3600)
    manager._postgres_repo = PostgresStateRepository(manager.postgres_adapter)
    manager._s3_repo = S3StateRepository(manager.s3_adapter, "test-bucket")

    with caplog.at_level("DEBUG"):
        keys = await manager.get_keys_by_pattern("order:*")

    assert keys == []
    assert any("Redis key lookup failed" in msg for msg in caplog.messages)
    assert any("PostgreSQL key lookup failed" in msg for msg in caplog.messages)
    assert any("S3 key lookup failed" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_get_storage_stats_handles_backend_errors(manager: sm.StateManager, caplog):
    from bot_v2.state.repositories import (
        RedisStateRepository,
        PostgresStateRepository,
        S3StateRepository,
    )

    # Re-assign to ensure all adapters are present for this test
    manager.redis_adapter = FailingRedisAdapter()
    manager.postgres_adapter = FailingPostgresAdapter()
    manager.s3_adapter = FailingS3Adapter()

    # Recreate repositories with failing adapters
    manager._redis_repo = RedisStateRepository(manager.redis_adapter, 3600)
    manager._postgres_repo = PostgresStateRepository(manager.postgres_adapter)
    manager._s3_repo = S3StateRepository(manager.s3_adapter, "test-bucket")

    with caplog.at_level("DEBUG"):
        stats = await manager.get_storage_stats()

    assert stats["hot_keys"] == 0
    assert stats["warm_keys"] == 0
    assert stats["cold_keys"] == 0
    assert any("Redis stats collection failed" in msg for msg in caplog.messages)
    assert any("PostgreSQL stats collection failed" in msg for msg in caplog.messages)
    assert any("S3 stats collection failed" in msg for msg in caplog.messages)
