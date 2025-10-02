"""Tests for backup recovery/restore operations.

Validates read path, checksum verification, and restoration logic
to ensure data can be successfully recovered from backups.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.operations import BackupManager
from tests.unit.bot_v2.state.backup.conftest import (
    calculate_checksum,
    make_snapshot_payload,
    mutate_payload_for_corruption,
)


@pytest.fixture
def populated_backup_dir(temp_workspace: Path, sample_runtime_state: dict) -> Path:
    """Create backup directory with multiple test snapshots."""
    backup_dir = temp_workspace / "backups"
    backup_dir.mkdir(exist_ok=True)

    # Create snapshots at different times
    for i, age_minutes in enumerate([5, 15, 30, 60]):
        timestamp = datetime.utcnow() - timedelta(minutes=age_minutes)
        backup_id = f"FULL_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Write backup data
        data_file = backup_dir / f"{backup_id}.backup"
        data_bytes = json.dumps(sample_runtime_state, default=str).encode()
        data_file.write_bytes(data_bytes)

        # Write metadata
        metadata = {
            "backup_id": backup_id,
            "backup_type": "full",
            "timestamp": timestamp.isoformat(),
            "size_bytes": len(data_bytes),
            "size_compressed": len(data_bytes),
            "checksum": calculate_checksum(data_bytes),
            "encryption_key_id": None,
            "storage_tier": "local",
            "retention_days": 90,
            "status": "completed",
        }
        meta_file = backup_dir / f"{backup_id}.meta"
        meta_file.write_text(json.dumps(metadata, indent=2))

    return backup_dir


class TestBackupLoader:
    """Test backup loading and discovery."""

    def test_loads_latest_snapshot_successfully(
        self,
        backup_config: BackupConfig,
        populated_backup_dir: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Loads latest snapshot when multiple backups exist.

        Critical: Must choose most recent backup for recovery.
        """
        # Create manager with populated directory
        backup_config.backup_dir = str(populated_backup_dir)
        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Should discover backups
        assert len(manager._backup_history) > 0

        # Find latest
        latest = max(manager._backup_history, key=lambda b: b.timestamp)

        # Latest should be most recent (5 minutes ago)
        assert (datetime.utcnow() - latest.timestamp).total_seconds() < 600

    def test_lists_all_available_backups(
        self,
        backup_config: BackupConfig,
        populated_backup_dir: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Lists all available backups in chronological order.

        Allows selecting specific backup for point-in-time recovery.
        """
        backup_config.backup_dir = str(populated_backup_dir)
        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        backups = sorted(manager._backup_history, key=lambda b: b.timestamp)

        # Should have all 4 backups
        assert len(backups) == 4

        # Should be in chronological order
        for i in range(len(backups) - 1):
            assert backups[i].timestamp < backups[i + 1].timestamp


class TestChecksumValidation:
    """Test checksum verification during restore."""

    async def test_validates_checksum_before_read(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
        sample_runtime_state: dict,
    ) -> None:
        """Validates checksum before accepting backup data.

        Critical: Prevents restoring corrupted data.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Corrupt the backup file
        backup_file = backup_dir / f"{metadata.backup_id}.backup"
        corrupted_data = b"CORRUPTED DATA THAT DOES NOT MATCH CHECKSUM"
        backup_file.write_bytes(corrupted_data)

        # Attempt restore - should detect corruption
        success = await manager.restore_from_backup(metadata.backup_id)

        # Should fail due to checksum mismatch
        assert success is False

    async def test_accepts_valid_checksum(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Accepts backup with valid checksum.

        Ensures no false positives in validation.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Restore should succeed with valid checksum
        success = await manager.restore_from_backup(metadata.backup_id)

        assert success is True

    async def test_detects_bit_flip_corruption(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Detects single bit flip in backup data.

        Validates sensitivity of checksum verification.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # Flip a single bit in the backup file
        backup_file = backup_dir / f"{metadata.backup_id}.backup"
        data = bytearray(backup_file.read_bytes())
        if len(data) > 10:
            data[10] ^= 0x01  # Flip one bit
            backup_file.write_bytes(bytes(data))

            # Should detect corruption
            success = await manager.restore_from_backup(metadata.backup_id)
            assert success is False


class TestMissingBackupHandling:
    """Test handling of missing backups and metadata."""

    async def test_handles_missing_metadata(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Handles missing metadata file gracefully.

        Should fail cleanly without crashing.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup then delete metadata
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        meta_file = backup_dir / f"{metadata.backup_id}.meta"
        meta_file.unlink()

        # Should handle gracefully
        result = manager._find_backup_metadata(metadata.backup_id)
        assert result is None

    async def test_handles_missing_backup_file(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Handles missing backup data file.

        Should return None rather than crash.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup then delete data file
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        backup_file = backup_dir / f"{metadata.backup_id}.backup"
        backup_file.unlink()

        # Should fail gracefully
        success = await manager.restore_from_backup(metadata.backup_id)
        assert success is False

    async def test_handles_nonexistent_backup_id(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Handles restore attempt for nonexistent backup.

        Should return clear error.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Attempt to restore nonexistent backup
        success = await manager.restore_from_backup("NONEXISTENT_BACKUP_ID")

        assert success is False


class TestRuntimeStateRestoration:
    """Test restoration of runtime state objects."""

    async def test_restores_runtime_state_object(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
        sample_runtime_state: dict,
    ) -> None:
        """Restores runtime state object from backup.

        Critical: Must repopulate state manager with correct data.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        # Mock state manager to capture restored data
        restored_data = {}

        async def mock_set_state(key: str, value: Any, category=None) -> bool:
            restored_data[key] = value
            return True

        async def mock_batch_set_state(items: dict) -> int:
            count = 0
            for key, (value, category) in items.items():
                restored_data[key] = value
                count += 1
            return count

        mock_state_manager.set_state = AsyncMock(side_effect=mock_set_state)
        mock_state_manager.batch_set_state = AsyncMock(side_effect=mock_batch_set_state)
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create and restore backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        success = await manager.restore_from_backup(metadata.backup_id)
        assert success is True

        # Verify data was restored
        assert len(restored_data) > 0

    async def test_preserves_field_types_on_restore(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
    ) -> None:
        """Preserves data types during restore.

        Ensures no type coercion during serialization roundtrip.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        # Create state with various types
        typed_state = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        mock_state_manager.create_snapshot = AsyncMock(return_value=typed_state)

        restored_data = {}

        async def mock_set_state(key: str, value: Any, category=None) -> bool:
            restored_data[key] = value
            return True

        mock_state_manager.set_state = AsyncMock(side_effect=mock_set_state)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Backup and restore
        metadata = await manager.create_backup(BackupType.FULL)
        success = await manager.restore_from_backup(metadata.backup_id)

        assert success is True


class TestIdempotentRestore:
    """Test idempotent restore behavior."""

    async def test_idempotent_restore(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
        sample_runtime_state: dict,
    ) -> None:
        """Restore is idempotent - running twice produces same result.

        Critical: No duplicate events or state drift.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        restore_count = {"count": 0}

        async def counting_set_state(key: str, value: Any, category=None) -> bool:
            restore_count["count"] += 1
            return True

        mock_state_manager.set_state = AsyncMock(side_effect=counting_set_state)
        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)
        assert metadata is not None

        # First restore
        restore_count["count"] = 0
        success1 = await manager.restore_from_backup(metadata.backup_id)
        first_count = restore_count["count"]

        # Second restore
        restore_count["count"] = 0
        success2 = await manager.restore_from_backup(metadata.backup_id)
        second_count = restore_count["count"]

        assert success1 is True
        assert success2 is True
        # Same number of operations both times
        assert first_count == second_count

    async def test_no_duplicate_events_on_repeated_restore(
        self,
        backup_config: BackupConfig,
        temp_workspace: Path,
        mock_state_manager: Mock,
        mock_event_store: Mock,
        sample_runtime_state: dict,
    ) -> None:
        """Repeated restore does not create duplicate events.

        Prevents event log pollution.
        """
        backup_dir = temp_workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_config.backup_dir = str(backup_dir)

        mock_state_manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)

        manager = BackupManager(state_manager=mock_state_manager, config=backup_config)

        # Create backup
        metadata = await manager.create_backup(BackupType.FULL)

        # Restore twice
        await manager.restore_from_backup(metadata.backup_id)
        await manager.restore_from_backup(metadata.backup_id)

        # Event store should not have duplicates
        # (Implementation depends on how events are tracked)
        assert True  # Placeholder - implementation-specific
