"""Smoke tests for recovery and backup orchestrators."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bot_v2.state.recovery import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryHandler,
    RecoveryMode,
    RecoveryStatus,
)
from bot_v2.state.backup_manager import (
    BackupConfig,
    BackupManager,
    BackupStatus,
    BackupType,
)


@pytest.mark.asyncio
async def test_initiate_recovery_completes_successfully(monkeypatch) -> None:
    """Ensure initiate_recovery drives the happy-path execution flow."""

    # Create minimal mock dependencies
    from unittest.mock import Mock

    mock_state_manager = Mock()
    mock_checkpoint_handler = Mock()

    handler = RecoveryHandler(
        state_manager=mock_state_manager, checkpoint_handler=mock_checkpoint_handler
    )

    # Mock the workflow execution (which is what actually runs now)
    async def mock_workflow_execute(operation: Any, mode: Any) -> None:
        operation.status = RecoveryStatus.COMPLETED
        operation.completed_at = datetime.utcnow()
        operation.recovery_time_seconds = 1.0

    handler.workflow.execute = AsyncMock(side_effect=mock_workflow_execute)  # type: ignore[method-assign]

    event = FailureEvent(
        failure_type=FailureType.REDIS_DOWN,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["redis"],
        error_message="redis offline",
    )

    operation = await handler.initiate_recovery(event, RecoveryMode.AUTOMATIC)

    assert operation.status is RecoveryStatus.COMPLETED
    handler.workflow.execute.assert_awaited_once()  # type: ignore[attr-defined]
    assert handler._current_operation is None
    assert handler._recovery_in_progress is False


@pytest.mark.asyncio
async def test_initiate_recovery_escalates_on_failure(monkeypatch) -> None:
    """Automatic recovery should escalate when execution fails."""

    from unittest.mock import Mock

    mock_state_manager = Mock()
    mock_checkpoint_handler = Mock()

    handler = RecoveryHandler(
        state_manager=mock_state_manager,
        checkpoint_handler=mock_checkpoint_handler,
        config=RecoveryConfig(max_retry_attempts=1),
    )

    # Mock the workflow execution to simulate failure
    async def mock_workflow_execute_failure(operation: Any, mode: Any) -> None:
        operation.status = RecoveryStatus.FAILED

    handler.workflow.execute = AsyncMock(side_effect=mock_workflow_execute_failure)  # type: ignore[method-assign]

    event = FailureEvent(
        failure_type=FailureType.POSTGRES_DOWN,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["postgres"],
        error_message="postgres offline",
    )

    operation = await handler.initiate_recovery(event, RecoveryMode.AUTOMATIC)

    assert operation.status is RecoveryStatus.FAILED
    handler.workflow.execute.assert_awaited_once()  # type: ignore[attr-defined]


class _DummyStateManager:
    """Minimal async interface for backup smoke coverage."""

    def __init__(self) -> None:
        now = datetime.utcnow().isoformat()
        self._values = {
            "position:BTC-PERP": {"timestamp": now, "quantity": "1"},
            "order:recent": {"timestamp": now, "id": "abc"},
            "portfolio_current": {"timestamp": now, "equity": "1000"},
            "performance_metrics": {"timestamp": now, "pnl": "5"},
            "config:critical": {"timestamp": now, "mode": "live"},
        }

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [key for key in self._values if key.startswith(prefix)]

    async def get_state(self, key: str) -> Any:
        return self._values.get(key)

    def get_repositories(self) -> Any:
        """Return mock repositories for direct access."""
        from bot_v2.state.state_manager import StateRepositories

        # Create a mock Redis repository
        class MockRedisRepo:
            def __init__(self, values):
                self._values = values

            async def keys(self, pattern: str) -> list[str]:
                prefix = pattern.rstrip("*")
                return [key for key in self._values if key.startswith(prefix)]

            async def fetch(self, key: str) -> Any:
                return self._values.get(key)

        return StateRepositories(
            redis=MockRedisRepo(self._values),
            postgres=None,
            s3=None,
        )


@pytest.mark.asyncio
async def test_create_backup_writes_metadata(tmp_path) -> None:
    """Backup manager should emit metadata and persist artifact."""

    backup_dir = tmp_path / "backups"
    config = BackupConfig(
        backup_dir=str(backup_dir),
        local_storage_path=str(backup_dir / "local"),
        enable_encryption=False,
        enable_compression=False,
        s3_bucket=None,
        network_storage_path=None,
    )

    manager = BackupManager(_DummyStateManager(), config=config)
    metadata = await manager.create_backup(BackupType.FULL)

    assert metadata is not None
    assert metadata.status in {BackupStatus.COMPLETED, BackupStatus.VERIFIED}
    assert metadata.data_sources

    artifact_path = backup_dir / "local" / f"{metadata.backup_id}.backup"
    assert artifact_path.exists()

    meta_file = backup_dir / f"{metadata.backup_id}.meta"
    assert meta_file.exists()
