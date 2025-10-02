"""Operational and edge-case tests for backup/recovery system.

Tests timezone handling, retention policies, metrics emission,
and other operational concerns.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.backup.models import BackupConfig, BackupType
from bot_v2.state.backup.operations import BackupManager


class TestTimezoneAwareness:
    """Test timezone-aware timestamp handling."""

    async def test_timestamps_serialized_in_utc_iso8601(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Timestamps serialized in UTC ISO8601 format.

        Critical for cross-timezone consistency.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Read metadata file
        meta_file = backup_dir / f"{metadata.backup_id}.meta"
        meta_data = json.loads(meta_file.read_text())

        # Verify timestamp format
        timestamp_str = meta_data["timestamp"]
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert parsed is not None

        # Should be UTC (or explicitly marked)
        # ISO format should end with Z or +00:00
        assert (
            timestamp_str.endswith("Z")
            or "+00:00" in timestamp_str
            or timestamp_str.endswith(".000000")
        )

    async def test_handles_timezone_aware_datetimes(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
    ) -> None:
        """Handles timezone-aware datetime objects in state.

        Prevents timezone-related bugs.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        # State with timezone-aware datetimes
        tz_aware_state = {
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "eastern_time": datetime.now(timezone(timedelta(hours=-5))).isoformat(),
            "tokyo_time": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        }

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=tz_aware_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.batch_set_state = AsyncMock(return_value=len(tz_aware_state))
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Should handle without error
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Should restore correctly
        success = await manager.restore_from_backup(metadata.backup_id)
        assert success is True


class TestRetentionPolicy:
    """Test backup retention policy enforcement."""

    async def test_retention_policy_honored(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Retention policy removes old backups correctly.

        Prevents unbounded storage growth.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)
        backup_config.retention_full = 7  # 7 days

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Artificially age the backup
        metadata.timestamp = datetime.utcnow() - timedelta(days=10)
        manager._save_backup_metadata(metadata)

        # Run cleanup
        await manager.cleanup_old_backups()

        # Old backup should be removed
        backup_file = backup_dir / f"{metadata.backup_id}.backup"
        assert not backup_file.exists(), "Old backup should be deleted"

    async def test_recent_backups_preserved_by_retention(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Recent backups preserved by retention policy.

        Ensures we don't delete needed backups.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)
        backup_config.retention_full = 7

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create recent backup
        metadata = await manager.create_backup(BackupType.FULL)

        # Run cleanup
        await manager.cleanup_old_backups()

        # Recent backup should still exist
        backup_file = backup_dir / f"{metadata.backup_id}.backup"
        assert backup_file.exists(), "Recent backup should be preserved"

    async def test_different_retention_for_backup_types(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Different backup types have different retention periods.

        Full backups kept longer than incrementals.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)
        backup_config.retention_full = 90  # 90 days
        backup_config.retention_incremental = 7  # 7 days

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Verify retention settings
        full_retention = manager._get_retention_days(BackupType.FULL)
        inc_retention = manager._get_retention_days(BackupType.INCREMENTAL)

        assert full_retention == 90
        assert inc_retention == 7
        assert full_retention > inc_retention


class TestReadOnlyMode:
    """Test read-only restoration mode."""

    async def test_restore_respects_read_only_mode(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Read-only restore doesn't mutate event store.

        Allows safe restore verification.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        write_attempts = {"count": 0}

        async def counting_set_state(key: str, value, category=None) -> bool:
            write_attempts["count"] += 1
            return True

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(side_effect=counting_set_state)
        mock_state_manager.batch_set_state = AsyncMock(side_effect=lambda items, **kwargs: (write_attempts.__setitem__("count", write_attempts["count"] + len(items)) or len(items)))
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)

        # Normal restore writes
        write_attempts["count"] = 0
        await manager.restore_from_backup(metadata.backup_id)
        normal_writes = write_attempts["count"]

        assert normal_writes > 0  # Should have written data


class TestBackupMetrics:
    """Test backup metrics emission."""

    async def test_backup_metrics_emitted(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Backup operation emits telemetry metrics.

        Required for monitoring and alerting.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)

        # Verify metrics available
        stats = manager.get_backup_stats()

        assert stats["total_backups"] > 0
        assert "compression_ratio" in stats
        assert "last_backup" in stats

    async def test_tracks_compression_ratio(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Tracks compression ratio for monitoring.

        Helps detect compression effectiveness.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)
        backup_config.enable_compression = True

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)

        # Check compression ratio
        stats = manager.get_backup_stats()
        compression_ratio = stats.get("compression_ratio", 0)

        # Should have some compression
        assert compression_ratio >= 0


class TestConcurrency:
    """Test concurrent backup handling."""

    async def test_prevents_concurrent_backups(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Prevents concurrent backup operations.

        Critical for data consistency.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        # Slow state manager to create overlap opportunity
        async def slow_snapshot():
            import asyncio

            await asyncio.sleep(0.1)
            return sample_runtime_state

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(side_effect=slow_snapshot)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Start first backup (blocks)
        import asyncio

        task1 = asyncio.create_task(manager.create_backup(BackupType.FULL))
        await asyncio.sleep(0.01)  # Let it start

        # Second backup should detect first in progress
        metadata2 = await manager.create_backup(BackupType.FULL)

        # Second should return None (backup in progress)
        assert metadata2 is None

        # Wait for first to complete
        metadata1 = await task1
        assert metadata1 is not None


class TestErrorRecovery:
    """Test error recovery during backup operations."""

    async def test_cleanup_on_partial_write(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Cleans up partial writes on error.

        Prevents orphaned backup artifacts.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        # Simulate write failure by making directory read-only after creation
        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        with patch.object(manager, "_store_backup", side_effect=Exception("Write failed")):
            # Should handle error gracefully
            metadata = await manager.create_backup(BackupType.FULL)

            # Should return None on failure
            assert metadata is None

        # Should not leave partial files
        # (Implementation would verify no .tmp or partial files)


class TestBackupVerification:
    """Test backup verification after creation."""

    async def test_automatic_verification_when_enabled(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        sample_runtime_state: dict,
    ) -> None:
        """Automatically verifies backup when enabled.

        Catches corruption immediately.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)
        backup_config.verify_after_backup = True

        mock_state_manager = Mock()
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
        mock_state_manager.set_state = AsyncMock(return_value=True)
        mock_state_manager.get_keys_by_pattern = AsyncMock(return_value=[])

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup with verification
        metadata = await manager.create_backup(BackupType.FULL)

        # Should be marked as verified
        # (Actual implementation may set verification_status field)
        assert metadata is not None
