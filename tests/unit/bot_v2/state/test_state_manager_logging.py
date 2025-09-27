"""Regression tests for StateManager logging pathways."""

from __future__ import annotations

import pytest

from bot_v2.state import state_manager as sm


@pytest.fixture
def manager(monkeypatch) -> sm.StateManager:
    """Provide a StateManager with external backends disabled."""

    monkeypatch.setattr(sm, "redis", None)
    monkeypatch.setattr(sm, "psycopg2", None)
    monkeypatch.setattr(sm, "boto3", None)

    return sm.StateManager(sm.StateConfig())


class _FailingRedis:
    def delete(self, key: str) -> None:  # noqa: D401 - simple stub
        raise RuntimeError("redis offline")

    def keys(self, pattern: str) -> list[str]:  # pragma: no cover - other tests use
        raise RuntimeError("redis pattern failure")

    def dbsize(self) -> int:  # pragma: no cover - other tests use
        raise RuntimeError("redis stats failure")


class _FailingCursor:
    def __enter__(self) -> "_FailingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, *args, **kwargs) -> None:
        raise RuntimeError("pg execute failure")

    def fetchall(self) -> list[dict]:  # pragma: no cover
        return []

    def fetchone(self) -> dict:  # pragma: no cover
        return {"count": 0}


class _FailingPostgres:
    def cursor(self) -> _FailingCursor:
        return _FailingCursor()

    def commit(self) -> None:  # pragma: no cover - commit should not be called after failure
        return None

    def rollback(self) -> None:
        raise RuntimeError("pg rollback failure")


class _FailingS3:
    def delete_object(self, **kwargs) -> None:
        raise RuntimeError("s3 delete failure")

    def list_objects_v2(self, **kwargs) -> dict:  # pragma: no cover
        raise RuntimeError("s3 list failure")

    def head_bucket(self, **kwargs) -> None:  # pragma: no cover
        raise RuntimeError("s3 head failure")


@pytest.mark.asyncio
async def test_delete_state_logs_backend_failures(manager: sm.StateManager, caplog):
    manager.redis_client = _FailingRedis()
    manager.pg_conn = _FailingPostgres()
    manager.s3_client = _FailingS3()

    with caplog.at_level("DEBUG"):
        success = await manager.delete_state("portfolio:test")

    assert success is False
    assert any("Failed to delete portfolio:test from Redis" in msg for msg in caplog.messages)
    assert any("Failed to delete portfolio:test from PostgreSQL" in msg for msg in caplog.messages)
    assert any("Failed to delete portfolio:test from S3" in msg for msg in caplog.messages)
    assert any("PostgreSQL rollback failed after delete error" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_get_keys_by_pattern_handles_backend_errors(manager: sm.StateManager, caplog):
    manager.redis_client = _FailingRedis()
    manager.pg_conn = _FailingPostgres()
    manager.s3_client = _FailingS3()

    with caplog.at_level("DEBUG"):
        keys = await manager.get_keys_by_pattern("order:*")

    assert keys == []
    assert any("Redis key lookup failed" in msg for msg in caplog.messages)
    assert any("PostgreSQL key lookup failed" in msg for msg in caplog.messages)
    assert any("S3 key lookup failed" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_get_storage_stats_handles_backend_errors(manager: sm.StateManager, caplog):
    manager.redis_client = _FailingRedis()
    manager.pg_conn = _FailingPostgres()
    manager.s3_client = _FailingS3()

    with caplog.at_level("DEBUG"):
        stats = await manager.get_storage_stats()

    assert stats["hot_keys"] == 0
    assert stats["warm_keys"] == 0
    assert stats["cold_keys"] == 0
    assert any("Redis stats collection failed" in msg for msg in caplog.messages)
    assert any("PostgreSQL stats collection failed" in msg for msg in caplog.messages)
    assert any("S3 stats collection failed" in msg for msg in caplog.messages)
