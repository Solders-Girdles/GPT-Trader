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

    handler = RecoveryHandler(state_manager=object(), checkpoint_handler=object())
    # Mock the new structure - alerter, validator
    handler.alerter.send_alert = AsyncMock()  # type: ignore[method-assign]
    handler._execute_recovery = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    handler.validator.validate_recovery = AsyncMock(return_value=True)  # type: ignore[method-assign]
    handler._cleanup_recovery_history = lambda: None  # type: ignore[assignment]

    event = FailureEvent(
        failure_type=FailureType.REDIS_DOWN,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["redis"],
        error_message="redis offline",
    )

    operation = await handler.initiate_recovery(event, RecoveryMode.AUTOMATIC)

    assert operation.status is RecoveryStatus.COMPLETED
    assert handler.alerter.send_alert.await_count == 2  # type: ignore[attr-defined]
    handler._execute_recovery.assert_awaited()  # type: ignore[attr-defined]
    handler.validator.validate_recovery.assert_awaited()  # type: ignore[attr-defined]
    assert handler._current_operation is None
    assert handler._recovery_in_progress is False


@pytest.mark.asyncio
async def test_initiate_recovery_escalates_on_failure(monkeypatch) -> None:
    """Automatic recovery should escalate when execution fails."""

    handler = RecoveryHandler(
        state_manager=object(),
        checkpoint_handler=object(),
        config=RecoveryConfig(max_retry_attempts=1),
    )
    # Mock the new structure
    handler.alerter.send_alert = AsyncMock()  # type: ignore[method-assign]
    handler._execute_recovery = AsyncMock(return_value=False)  # type: ignore[attr-defined]
    handler.validator.validate_recovery = AsyncMock(return_value=False)  # type: ignore[method-assign]
    handler._cleanup_recovery_history = lambda: None  # type: ignore[assignment]
    handler.alerter.escalate_recovery = AsyncMock()  # type: ignore[method-assign]

    event = FailureEvent(
        failure_type=FailureType.POSTGRES_DOWN,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["postgres"],
        error_message="postgres offline",
    )

    operation = await handler.initiate_recovery(event, RecoveryMode.AUTOMATIC)

    assert operation.status is RecoveryStatus.FAILED
    handler.alerter.escalate_recovery.assert_awaited_once()  # type: ignore[attr-defined]


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
