from __future__ import annotations

from collections.abc import Callable
import gc
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_v2.state.recovery.handlers.system import SystemRecoveryHandlers
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)
from bot_v2.state.state_manager import StateCategory


@pytest.fixture
def operation_factory() -> Callable[[], RecoveryOperation]:
    def _factory() -> RecoveryOperation:
        now = datetime.utcnow()
        event = FailureEvent(
            failure_type=FailureType.API_GATEWAY_DOWN,
            timestamp=now,
            severity="critical",
            affected_components=["system"],
            error_message="failure",
        )
        return RecoveryOperation(
            operation_id="op-test",
            failure_event=event,
            recovery_mode=RecoveryMode.AUTOMATIC,
            status=RecoveryStatus.IN_PROGRESS,
            started_at=now,
        )

    return _factory


@pytest.mark.asyncio
async def test_recover_from_memory_overflow_demotes_hot_keys(monkeypatch, operation_factory):
    operation = operation_factory()

    class StubStateManager:
        def __init__(self) -> None:
            self._local_cache = {"a": 1}
            self.demoted: list[str] = []

        async def get_keys_by_pattern(self, pattern: str) -> list[str]:
            assert pattern == "*"
            return [f"key-{i}" for i in range(120)]

        async def demote_to_cold(self, key: str) -> bool:
            self.demoted.append(key)
            return True

    state_manager = StubStateManager()
    handler = SystemRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    collected = {"called": False}
    monkeypatch.setattr(gc, "collect", lambda: collected.__setitem__("called", True))

    result = await handler.recover_from_memory_overflow(operation)

    assert result is True
    assert state_manager._local_cache == {}
    assert len(state_manager.demoted) == 100
    assert collected["called"] is True
    assert operation.actions_taken == [
        "Starting memory recovery",
        "Cleared local cache",
        "Demoted 100 keys to cold storage",
        "Triggered garbage collection",
    ]


@pytest.mark.asyncio
async def test_recover_from_memory_overflow_handles_errors(operation_factory):
    operation = operation_factory()

    class FaultyStateManager:
        def __init__(self) -> None:
            self._local_cache: dict[str, int] = {}

        async def get_keys_by_pattern(self, pattern: str) -> list[str]:  # noqa: ARG002
            raise RuntimeError("cache failure")

    handler = SystemRecoveryHandlers(FaultyStateManager(), checkpoint_handler=SimpleNamespace())

    result = await handler.recover_from_memory_overflow(operation)

    assert result is False
    assert operation.actions_taken == [
        "Starting memory recovery",
        "Cleared local cache",
    ]


@pytest.mark.asyncio
async def test_recover_from_disk_full_removes_temp_files(monkeypatch, tmp_path, operation_factory):
    operation = operation_factory()
    cleanup_called = {"flag": False}

    class DummyCheckpoint:
        def _cleanup_old_checkpoints(self) -> None:
            cleanup_called["flag"] = True

    bot_temp = tmp_path / "bot_v2"
    bot_temp.mkdir()

    removed_paths: list[str] = []
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "shutil.rmtree", lambda path, ignore_errors=True: removed_paths.append(path)
    )

    handler = SystemRecoveryHandlers(SimpleNamespace(), DummyCheckpoint())
    result = await handler.recover_from_disk_full(operation)

    assert result is True
    assert cleanup_called["flag"] is True
    assert str(bot_temp) in removed_paths
    assert operation.actions_taken == [
        "Starting disk space recovery",
        "Cleaned old checkpoints",
        "Cleared temporary files",
    ]


@pytest.mark.asyncio
async def test_recover_from_disk_full_handles_errors(operation_factory):
    operation = operation_factory()

    class FailingCheckpoint:
        def _cleanup_old_checkpoints(self) -> None:
            raise RuntimeError("boom")

    handler = SystemRecoveryHandlers(SimpleNamespace(), FailingCheckpoint())
    result = await handler.recover_from_disk_full(operation)

    assert result is False
    assert operation.actions_taken == ["Starting disk space recovery"]


@pytest.mark.asyncio
async def test_recover_from_network_partition_reinitializes_services(
    monkeypatch, operation_factory
):
    operation = operation_factory()

    state_manager = SimpleNamespace(
        _init_redis=MagicMock(),
        _init_postgres=MagicMock(),
    )

    handler = SystemRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    sleep_spy = AsyncMock()
    sync_spy = AsyncMock()
    monkeypatch.setattr("bot_v2.state.recovery.handlers.system.asyncio.sleep", sleep_spy)
    monkeypatch.setattr(SystemRecoveryHandlers, "_synchronize_state", sync_spy)

    result = await handler.recover_from_network_partition(operation)

    assert result is True
    sleep_spy.assert_awaited_once_with(5)
    state_manager._init_redis.assert_called_once()
    state_manager._init_postgres.assert_called_once()
    sync_spy.assert_awaited_once_with()
    assert operation.actions_taken == [
        "Handling network partition",
        "Re-established Redis connection",
        "Re-established PostgreSQL connection",
        "Synchronized distributed state",
    ]


@pytest.mark.asyncio
async def test_recover_from_network_partition_handles_failure(monkeypatch, operation_factory):
    operation = operation_factory()

    handler = SystemRecoveryHandlers(SimpleNamespace(), checkpoint_handler=SimpleNamespace())

    sleep_spy = AsyncMock()
    failing_sync = AsyncMock(side_effect=RuntimeError("sync failed"))
    monkeypatch.setattr("bot_v2.state.recovery.handlers.system.asyncio.sleep", sleep_spy)
    monkeypatch.setattr(SystemRecoveryHandlers, "_synchronize_state", failing_sync)

    result = await handler.recover_from_network_partition(operation)

    assert result is False
    sleep_spy.assert_awaited_once_with(5)
    assert operation.actions_taken == ["Handling network partition"]


@pytest.mark.asyncio
async def test_recover_api_gateway_resets_rate_limits(operation_factory):
    operation = operation_factory()

    class ApiStateManager:
        def __init__(self) -> None:
            self.set_calls: list[tuple[str, str]] = []
            self.deleted: list[str] = []

        async def set_state(self, key: str, value: str, category=None) -> None:  # noqa: ANN001
            self.set_calls.append((key, value))

        async def get_keys_by_pattern(self, pattern: str) -> list[str]:
            assert pattern == "rate_limit:*"
            return ["rate_limit:BTC", "rate_limit:ETH"]

        async def delete_state(self, key: str) -> None:
            self.deleted.append(key)

        async def batch_delete_state(self, keys: list[str]) -> int:
            self.deleted.extend(keys)
            return len(keys)

    state_manager = ApiStateManager()
    handler = SystemRecoveryHandlers(state_manager, checkpoint_handler=SimpleNamespace())

    result = await handler.recover_api_gateway(operation)

    assert result is True
    assert state_manager.set_calls == [
        ("system:api_gateway_status", "restarting"),
        ("system:api_gateway_status", "recovered"),
    ]
    assert state_manager.deleted == ["rate_limit:BTC", "rate_limit:ETH"]
    assert operation.actions_taken == [
        "Starting API gateway recovery",
        "Cleared 2 rate limit counters",
        "API gateway recovery completed",
    ]


@pytest.mark.asyncio
async def test_recover_api_gateway_handles_errors(operation_factory):
    operation = operation_factory()

    class FaultyManager:
        async def set_state(self, key: str, value: str, category=None) -> None:  # noqa: ANN001
            raise RuntimeError("unavailable")

    handler = SystemRecoveryHandlers(FaultyManager(), checkpoint_handler=SimpleNamespace())

    result = await handler.recover_api_gateway(operation)

    assert result is False
    assert operation.actions_taken == ["Starting API gateway recovery"]


@pytest.mark.asyncio
async def test_synchronize_state_promotes_positions(monkeypatch):
    class SyncStateManager:
        def __init__(self) -> None:
            self.batch_set_calls: list[dict[str, tuple]] = []

        async def get_keys_by_pattern(self, pattern: str) -> list[str]:
            assert pattern == "position:*"
            return ["position:BTC"]

        async def get_state(self, key: str) -> dict[str, int]:
            return {"size": 3}

        async def batch_set_state(self, items: dict, ttl_seconds=None) -> int:  # noqa: ANN001
            self.batch_set_calls.append(items)
            return len(items)

    handler = SystemRecoveryHandlers(SyncStateManager(), checkpoint_handler=SimpleNamespace())
    await handler._synchronize_state()

    assert len(handler.state_manager.batch_set_calls) == 1
    batch_items = handler.state_manager.batch_set_calls[0]
    assert "position:BTC" in batch_items
    assert batch_items["position:BTC"] == ({"size": 3}, StateCategory.HOT)


@pytest.mark.asyncio
async def test_synchronize_state_logs_errors(caplog):
    class FaultyStateManager:
        async def get_keys_by_pattern(self, pattern: str) -> list[str]:
            raise RuntimeError("network")

    handler = SystemRecoveryHandlers(FaultyStateManager(), checkpoint_handler=SimpleNamespace())

    with caplog.at_level("ERROR"):
        await handler._synchronize_state()

    assert any("State synchronization failed" in msg for msg in caplog.messages)
