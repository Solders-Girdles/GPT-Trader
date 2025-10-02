"""Tests for storage recovery handlers."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_v2.state.recovery.handlers.storage import StorageRecoveryHandlers
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)


def make_operation() -> RecoveryOperation:
    now = datetime.utcnow()
    event = FailureEvent(
        failure_type=FailureType.POSTGRES_DOWN,
        timestamp=now,
        severity="critical",
        affected_components=["postgres"],
        error_message="offline",
    )
    return RecoveryOperation(
        operation_id="op-storage",
        failure_event=event,
        recovery_mode=RecoveryMode.AUTOMATIC,
        status=RecoveryStatus.IN_PROGRESS,
        started_at=now,
    )


@pytest.mark.asyncio
async def test_recover_postgres_restores_from_checkpoint() -> None:
    operation = make_operation()

    timestamp = datetime.utcnow() - timedelta(seconds=42)
    checkpoint = SimpleNamespace(checkpoint_id="cp-001", timestamp=timestamp)

    checkpoint_handler = SimpleNamespace(
        get_latest_checkpoint=lambda: checkpoint,
        restore_from_checkpoint=AsyncMock(return_value=True),
    )

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler)

    result = await handler.recover_postgres(operation)

    assert result is True
    assert operation.actions_taken[-1] == "Restored from checkpoint cp-001"
    assert "42" in operation.data_loss_estimate


@pytest.mark.asyncio
async def test_recover_postgres_uses_backup_when_no_checkpoint() -> None:
    operation = make_operation()

    restore_latest_backup = AsyncMock(return_value=True)
    checkpoint_handler = SimpleNamespace(get_latest_checkpoint=lambda: None)
    backup_manager = SimpleNamespace(restore_latest_backup=restore_latest_backup)

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler, backup_manager)

    result = await handler.recover_postgres(operation)

    assert result is True
    restore_latest_backup.assert_awaited_once()
    assert operation.actions_taken == ["Starting PostgreSQL recovery from checkpoint"]


@pytest.mark.asyncio
async def test_recover_postgres_no_checkpoint_no_backup() -> None:
    operation = make_operation()

    handler = StorageRecoveryHandlers(
        SimpleNamespace(),
        SimpleNamespace(get_latest_checkpoint=lambda: None),
        backup_manager=None,
    )

    result = await handler.recover_postgres(operation)

    assert result is False
    assert operation.actions_taken == ["Starting PostgreSQL recovery from checkpoint"]


@pytest.mark.asyncio
async def test_recover_s3_sets_local_fallback() -> None:
    operation = make_operation()

    set_state = AsyncMock()
    state_manager = SimpleNamespace(set_state=set_state)
    handler = StorageRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_s3(operation)

    assert result is True
    set_state.assert_awaited_once_with("system:s3_available", False)
    assert operation.actions_taken == [
        "S3 recovery - using local storage fallback",
        "Configured local disk fallback for cold storage",
    ]


@pytest.mark.asyncio
async def test_recover_s3_handles_errors() -> None:
    operation = make_operation()

    state_manager = SimpleNamespace(set_state=AsyncMock(side_effect=RuntimeError("redis down")))
    handler = StorageRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_s3(operation)

    assert result is False
    assert operation.actions_taken == ["S3 recovery - using local storage fallback"]


@pytest.mark.asyncio
async def test_recover_from_corruption_restores_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = make_operation()

    timestamp = datetime.utcnow() - timedelta(minutes=1)
    checkpoint = SimpleNamespace(checkpoint_id="cp-42", timestamp=timestamp)

    checkpoint_handler = SimpleNamespace(
        find_valid_checkpoint=AsyncMock(return_value=checkpoint),
        restore_from_checkpoint=AsyncMock(return_value=True),
    )

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler)
    replay_spy = AsyncMock(return_value=True)
    monkeypatch.setattr(StorageRecoveryHandlers, "_replay_transactions_from", replay_spy)

    result = await handler.recover_from_corruption(operation)

    assert result is True
    assert operation.actions_taken[-2:] == [
        "Restored from valid checkpoint cp-42",
        "Replayed transactions from checkpoint",
    ]
    replay_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_from_corruption_no_checkpoint() -> None:
    operation = make_operation()

    checkpoint_handler = SimpleNamespace(
        find_valid_checkpoint=AsyncMock(return_value=None),
        restore_from_checkpoint=AsyncMock(return_value=True),
    )

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler)

    result = await handler.recover_from_corruption(operation)

    assert result is False
    assert operation.actions_taken == ["Starting corruption recovery"]


@pytest.mark.asyncio
async def test_recover_from_corruption_restore_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = make_operation()

    checkpoint = SimpleNamespace(checkpoint_id="cp-fail", timestamp=datetime.utcnow())

    checkpoint_handler = SimpleNamespace(
        find_valid_checkpoint=AsyncMock(return_value=checkpoint),
        restore_from_checkpoint=AsyncMock(return_value=False),
    )

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler)
    monkeypatch.setattr(
        StorageRecoveryHandlers, "_replay_transactions_from", AsyncMock(return_value=False)
    )

    result = await handler.recover_from_corruption(operation)

    assert result is False
    assert operation.actions_taken == ["Starting corruption recovery"]


@pytest.mark.asyncio
async def test_replay_transactions_success() -> None:
    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler=SimpleNamespace())

    result = await handler._replay_transactions_from(datetime.utcnow())

    assert result is True


@pytest.mark.asyncio
async def test_recover_from_corruption_handles_exceptions() -> None:
    operation = make_operation()

    checkpoint_handler = SimpleNamespace(
        find_valid_checkpoint=AsyncMock(side_effect=RuntimeError("io error")),
        restore_from_checkpoint=AsyncMock(return_value=True),
    )

    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler)

    result = await handler.recover_from_corruption(operation)

    assert result is False
    assert operation.actions_taken[-1] == "Corruption recovery error: io error"


async def test_recover_redis_restores_hot_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = make_operation()

    class FakeCursor:
        def __init__(self) -> None:
            self.rows = [
                {"key": "hot:1", "data": {"value": 1}},
                {"key": "hot:2", "data": {"value": 2}},
            ]

        def execute(self, query: str, params: tuple) -> None:  # noqa: ARG002
            assert "state_warm" in query

        def fetchall(self) -> list[dict[str, object]]:
            return self.rows

    fake_cursor = FakeCursor()

    class CursorContext:
        def __enter__(self) -> FakeCursor:
            return fake_cursor

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    class PgConn:
        def cursor(self) -> CursorContext:
            return CursorContext()

    class StateManagerStub:
        def __init__(self):
            self.pg_conn = PgConn()
            self.batch_set_items = []

        async def batch_set_state(self, items: dict, ttl_seconds=None) -> int:  # noqa: ANN001
            self.batch_set_items.append(items)
            # Simulate partial failure on hot:1
            return len([k for k in items.keys() if k != "hot:1"])

    state_manager = StateManagerStub()
    handler = StorageRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_redis(operation)

    assert result is True
    # Verify batch_set_state was called with both items
    assert len(state_manager.batch_set_items) == 1
    items = state_manager.batch_set_items[0]
    assert "hot:1" in items
    assert "hot:2" in items
    assert items["hot:2"][0] == {"value": 2}
    assert operation.actions_taken == [
        "Starting Redis recovery from PostgreSQL",
        "Recovered 1 keys to Redis",  # Only 1 succeeded (hot:2)
    ]


@pytest.mark.asyncio
async def test_recover_redis_without_pg_conn_returns_false() -> None:
    operation = make_operation()
    state_manager = SimpleNamespace(pg_conn=None, redis_client=SimpleNamespace())
    handler = StorageRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_redis(operation)

    assert result is False
    assert operation.actions_taken == ["Starting Redis recovery from PostgreSQL"]


@pytest.mark.asyncio
async def test_recover_redis_captures_errors() -> None:
    operation = make_operation()

    class PgConn:
        def cursor(self):
            raise RuntimeError("cursor blown up")

    state_manager = SimpleNamespace(pg_conn=PgConn(), redis_client=SimpleNamespace())
    handler = StorageRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_redis(operation)

    assert result is False
    assert operation.actions_taken[-1] == "Redis recovery error: cursor blown up"


@pytest.mark.asyncio
async def test_recover_postgres_handles_exceptions() -> None:
    operation = make_operation()

    class FaultyCheckpointHandler:
        def get_latest_checkpoint(self):
            raise RuntimeError("checkpoint service down")

    handler = StorageRecoveryHandlers(
        state_manager=SimpleNamespace(),
        checkpoint_handler=FaultyCheckpointHandler(),
    )

    result = await handler.recover_postgres(operation)

    assert result is False
    assert operation.actions_taken == [
        "Starting PostgreSQL recovery from checkpoint",
        "PostgreSQL recovery error: checkpoint service down",
    ]


@pytest.mark.asyncio
async def test_replay_transactions_handles_logging_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = StorageRecoveryHandlers(SimpleNamespace(), checkpoint_handler=SimpleNamespace())

    def boom(*_args, **_kwargs):
        raise RuntimeError("logging failed")

    monkeypatch.setattr("bot_v2.state.recovery.handlers.storage.logger.info", boom)

    result = await handler._replay_transactions_from(datetime.utcnow())

    assert result is False
