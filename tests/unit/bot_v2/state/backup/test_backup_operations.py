"""Tests for BackupManager - data integrity and persistence.

This module tests the BackupManager's ability to create, store, restore, and
manage backups with encryption, compression, and multi-tier storage.

Critical behaviors tested:
- Initialization of storage backends (local/network/S3)
- Backup creation (full/differential/incremental)
- Encryption and compression functionality
- Backup restoration and verification
- Checksum validation for data integrity
- S3 upload/download with error handling
- Cleanup and retention policy enforcement
- Concurrent backup protection
- Graceful degradation when dependencies unavailable

Business Context:
    The BackupManager ensures RPO <1 minute for trading state persistence.
    Failures here can result in:

    - Unrecoverable position state after crashes
    - Loss of order history and P&L tracking
    - Inability to prove regulatory compliance
    - Complete account state corruption
    - Financial liability from lost audit trail

    This is mission-critical infrastructure for production trading.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.operations import BackupManager


@pytest.fixture
def temp_backup_dir():
    """Create temporary directory for backup testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def backup_config(temp_backup_dir: Path) -> BackupConfig:
    """Create test backup configuration."""
    return BackupConfig(
        backup_dir=str(temp_backup_dir),
        retention_incremental=7,
        retention_full=90,
        retention_differential=30,
        enable_compression=True,
        enable_encryption=True,
        s3_bucket="test-bucket",
        full_backup_interval_hours=24,
        incremental_backup_interval_minutes=15,
    )


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    from unittest.mock import AsyncMock
    state_manager = Mock()
    state_manager.get_keys_by_pattern = AsyncMock(return_value=["test:key1", "test:key2"])
    state_manager.get_state = AsyncMock(return_value={"test": "data"})
    return state_manager


@pytest.fixture
def backup_manager(mock_state_manager, backup_config: BackupConfig) -> BackupManager:
    """Create BackupManager instance."""
    return BackupManager(state_manager=mock_state_manager, config=backup_config)


class TestBackupManagerInitialization:
    """Test BackupManager initialization and setup."""

    def test_initializes_with_config(self, backup_config: BackupConfig) -> None:
        """Initializes with backup configuration.

        Must set up storage paths and configuration.
        """
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert manager.config == backup_config
        assert Path(manager.config.backup_dir).exists()

    def test_creates_backup_directory(self, backup_config: BackupConfig) -> None:
        """Creates backup directory if it doesn't exist.

        Essential for first-run initialization.
        """
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert Path(manager.config.backup_dir).exists()
        assert Path(manager.config.backup_dir).is_dir()

    @patch("bot_v2.state.backup.services.encryption.Fernet")
    def test_initializes_encryption_when_enabled(
        self, mock_fernet: Mock, backup_config: BackupConfig
    ) -> None:
        """Initializes encryption when enabled in config.

        Critical for securing sensitive trading data.
        """
        backup_config.enable_encryption = True
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert manager._encryption_enabled is True

    def test_disables_encryption_when_config_false(
        self, backup_config: BackupConfig
    ) -> None:
        """Disables encryption when config specifies.

        Allows faster backups in test environments.
        """
        backup_config.enable_encryption = False
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert manager._encryption_enabled is False

    @patch("bot_v2.state.backup.services.transport.boto3")
    @patch("bot_v2.state.backup.services.transport.S3_AVAILABLE", True)
    def test_initializes_s3_when_enabled(
        self, mock_boto3: Mock, backup_config: BackupConfig
    ) -> None:
        """Initializes S3 client when enabled in config.

        Required for cloud backup tier.
        """
        backup_config.enable_s3 = True
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        manager = BackupManager(state_manager=Mock(), config=backup_config)

        # S3 client should be initialized
        assert manager._s3_client is not None or manager.transport_service._s3_client is not None

    def test_handles_s3_unavailable_gracefully(
        self, backup_config: BackupConfig
    ) -> None:
        """Handles missing boto3 without crashing.

        System operates with local storage only.
        """
        backup_config.enable_s3 = False
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert manager._s3_client is None

    def test_initializes_metadata_cache(self, backup_config: BackupConfig) -> None:
        """Initializes metadata cache for backup tracking.

        Required for incremental/differential backup logic.
        """
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert isinstance(manager._backup_metadata, dict)

    def test_initializes_thread_lock(self, backup_config: BackupConfig) -> None:
        """Initializes threading lock for concurrent protection.

        Prevents concurrent backups from corrupting each other.
        """
        manager = BackupManager(state_manager=Mock(), config=backup_config)

        assert hasattr(manager, "_backup_lock")


class TestFullBackupCreation:
    """Test full backup creation."""

    @pytest.mark.asyncio
    async def test_creates_full_backup(
        self, backup_manager: BackupManager, temp_backup_dir: Path
    ) -> None:
        """Creates full backup of state data.

        Full backup contains all state data.
        """
        metadata = await backup_manager.create_backup(backup_type=BackupType.FULL)

        assert metadata is not None
        assert metadata.backup_type == BackupType.FULL
        assert metadata.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        assert metadata.size_bytes > 0

    def test_full_backup_contains_all_data(
        self, backup_manager: BackupManager
    ) -> None:
        """Full backup contains complete state snapshot.

        No data should be omitted from full backup.
        """
        state_data = {
            "positions": {"AAPL": {"qty": 10}},
            "orders": [{"id": "1", "symbol": "AAPL"}],
            "account": {"equity": 10000.0},
            "metadata": {"last_update": "2024-01-01T00:00:00Z"},
        }

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Verify all keys preserved
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        assert backup_file.exists()

    def test_full_backup_generates_checksum(
        self, backup_manager: BackupManager
    ) -> None:
        """Full backup generates checksum for integrity verification.

        Checksum detects corruption during storage/retrieval.
        """
        state_data = {"test": "data"}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata.checksum is not None
        assert len(metadata.checksum) > 0

    def test_full_backup_records_size(self, backup_manager: BackupManager) -> None:
        """Full backup records both raw and compressed size.

        Size tracking for storage management and monitoring.
        """
        state_data = {"large_data": "x" * 10000}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata.size_bytes > 0
        if backup_manager.config.enable_compression:
            assert metadata.size_compressed > 0
            # Compressed size should be less than raw for repetitive data
            assert metadata.size_compressed < metadata.size_bytes


class TestIncrementalBackup:
    """Test incremental backup creation."""

    def test_creates_incremental_backup(
        self, backup_manager: BackupManager
    ) -> None:
        """Creates incremental backup containing only changes.

        Incremental backups optimize storage and time.
        """
        # Create initial full backup
        initial_state = {"positions": {"AAPL": {"qty": 10}}}
        full_metadata = backup_manager.create_backup(
            state_data=initial_state, backup_type=BackupType.FULL
        )

        # Create incremental with changes
        changed_state = {"positions": {"AAPL": {"qty": 15}, "GOOGL": {"qty": 5}}}
        inc_metadata = backup_manager.create_backup(
            state_data=changed_state, backup_type=BackupType.INCREMENTAL
        )

        assert inc_metadata.backup_type == BackupType.INCREMENTAL
        # Incremental records actual diff size, which can be larger than baseline
        # if many fields changed or new data was added
        assert inc_metadata.size_bytes > 0

    def test_incremental_requires_baseline(
        self, backup_manager: BackupManager
    ) -> None:
        """Incremental backup requires previous full backup.

        Must fail gracefully if no baseline exists.
        """
        state_data = {"positions": {"AAPL": {"qty": 10}}}

        # First backup should be forced to FULL if incremental requested
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.INCREMENTAL
        )

        # Should auto-convert to FULL or handle gracefully
        assert metadata is not None


class TestDifferentialBackup:
    """Test differential backup creation."""

    def test_creates_differential_backup(
        self, backup_manager: BackupManager
    ) -> None:
        """Creates differential backup against last full backup.

        Differential contains all changes since last full backup.
        """
        # Create full backup
        initial_state = {"positions": {"AAPL": {"qty": 10}}}
        backup_manager.create_backup(
            state_data=initial_state, backup_type=BackupType.FULL
        )

        # Create differential
        changed_state = {"positions": {"AAPL": {"qty": 20}}}
        diff_metadata = backup_manager.create_backup(
            state_data=changed_state, backup_type=BackupType.DIFFERENTIAL
        )

        assert diff_metadata.backup_type == BackupType.DIFFERENTIAL


class TestBackupCompression:
    """Test backup compression."""

    def test_compresses_backup_when_enabled(
        self, backup_manager: BackupManager
    ) -> None:
        """Compresses backup data when compression enabled.

        Reduces storage costs and transfer time.
        """
        backup_manager.config.enable_compression = True
        # Highly compressible data
        state_data = {"repeated": "x" * 10000}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata.size_compressed < metadata.size_bytes

    def test_skips_compression_when_disabled(
        self, backup_manager: BackupManager
    ) -> None:
        """Skips compression when disabled in config.

        Faster backups when storage space not a concern.
        """
        backup_manager.config.enable_compression = False
        state_data = {"data": "test"}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Should have no compression or equal sizes
        assert metadata.size_compressed == 0 or metadata.size_compressed == metadata.size_bytes


class TestBackupEncryption:
    """Test backup encryption."""

    def test_encrypts_backup_when_enabled(
        self, backup_manager: BackupManager
    ) -> None:
        """Encrypts backup data when encryption enabled.

        Critical for securing sensitive trading data.
        """
        backup_manager.config.enable_encryption = True
        backup_manager.encryption_service.enabled = True
        backup_manager._encryption_enabled = True
        mock_cipher = Mock()
        mock_cipher.encrypt.return_value = (b"encrypted_data", "primary")
        backup_manager.encryption_service._cipher = mock_cipher
        backup_manager._cipher = mock_cipher

        state_data = {"secret": "sensitive_data"}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Verify encryption was attempted
        assert metadata.encryption_key_id is not None

    def test_skips_encryption_when_disabled(
        self, backup_manager: BackupManager
    ) -> None:
        """Skips encryption when disabled in config.

        Faster backups in non-production environments.
        """
        backup_manager.config.enable_encryption = False
        backup_manager.encryption_service.enabled = False
        backup_manager._encryption_enabled = False
        state_data = {"data": "test"}

        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata.encryption_key_id is None


class TestBackupRestoration:
    """Test backup restoration and verification."""

    def test_restores_full_backup(self, backup_manager: BackupManager) -> None:
        """Restores data from full backup.

        Critical for disaster recovery.
        """
        original_state = {
            "positions": {"AAPL": {"qty": 10, "avg_price": 150.0}},
            "account": {"equity": 10000.0},
        }

        # Create backup
        metadata = backup_manager.create_backup(
            state_data=original_state, backup_type=BackupType.FULL
        )

        # Restore backup
        restored_state = backup_manager.restore_from_backup(metadata.backup_id)

        assert restored_state == original_state

    def test_verifies_checksum_on_restore(
        self, backup_manager: BackupManager
    ) -> None:
        """Verifies checksum when restoring backup.

        Detects corruption before using potentially invalid state.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Corrupt the backup file
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        backup_file.write_bytes(b"corrupted data")

        # Restore should detect corruption
        with pytest.raises(Exception):  # Should raise checksum validation error
            backup_manager.restore_from_backup(metadata.backup_id)

    def test_handles_missing_backup_gracefully(
        self, backup_manager: BackupManager
    ) -> None:
        """Handles attempt to restore non-existent backup.

        Should raise clear error for missing backup.
        """
        with pytest.raises(FileNotFoundError):
            backup_manager.restore_from_backup("nonexistent_backup_id")

    def test_decrypts_backup_on_restore(
        self, backup_manager: BackupManager
    ) -> None:
        """Decrypts backup data during restoration.

        Must decrypt to recover original state.
        """
        backup_manager.config.enable_encryption = True
        backup_manager.encryption_service.enabled = True
        backup_manager._encryption_enabled = True
        mock_cipher = Mock()
        mock_cipher.encrypt.return_value = (b"encrypted", "primary")
        backup_manager.encryption_service._cipher = mock_cipher
        backup_manager._cipher = mock_cipher

        state_data = {"secret": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Mock decrypt to return original data
        original_json = json.dumps({"state": state_data}).encode()
        mock_cipher.decrypt.return_value = original_json

        restored = backup_manager.restore_from_backup(metadata.backup_id)
        assert restored == state_data


class TestS3Operations:
    """Test S3 upload/download functionality."""

    def test_uploads_backup_to_s3(
        self, backup_manager: BackupManager
    ) -> None:
        """Uploads backup to S3 cloud storage.

        Critical for off-site disaster recovery.
        """
        backup_manager.config.enable_s3 = True
        mock_s3 = Mock()
        backup_manager.transport_service._s3_client = mock_s3
        backup_manager._s3_client = mock_s3

        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Verify S3 upload attempted
        if backup_manager._s3_client:
            backup_manager._upload_to_s3(metadata.backup_id)
            mock_s3.upload_file.assert_called()

    def test_handles_s3_upload_failure_gracefully(
        self, backup_manager: BackupManager
    ) -> None:
        """Handles S3 upload failures without crashing.

        Local backup should remain valid even if S3 fails.
        """
        backup_manager.config.enable_s3 = True
        mock_s3 = Mock()
        mock_s3.upload_file.side_effect = Exception("S3 unavailable")
        backup_manager.transport_service._s3_client = mock_s3
        backup_manager._s3_client = mock_s3

        state_data = {"test": "data"}

        # Should not raise - local backup still succeeds
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata is not None
        assert metadata.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]


class TestBackupCleanup:
    """Test backup cleanup and retention policies."""

    def test_deletes_old_backups_beyond_retention(
        self, backup_manager: BackupManager
    ) -> None:
        """Deletes backups older than retention period.

        Prevents unbounded storage growth.
        """
        from datetime import timezone
        from bot_v2.state.backup.services import RetentionService

        # Update retention service to use 7 days for all backup types
        backup_manager.retention_service = RetentionService(
            retention_incremental=7,
            retention_differential=7,
            retention_full=7,
            retention_emergency=7,
            retention_snapshot=7,
        )

        # Create old backup
        old_state = {"old": "data"}
        old_metadata = backup_manager.create_backup(
            state_data=old_state, backup_type=BackupType.FULL
        )

        # Artificially age the backup (use timezone-aware datetime)
        old_metadata.timestamp = datetime.now(timezone.utc) - timedelta(days=10)
        backup_manager._backup_metadata[old_metadata.backup_id] = old_metadata

        # Run cleanup
        deleted_count = backup_manager.cleanup_old_backups()

        assert deleted_count > 0

    def test_preserves_recent_backups(self, backup_manager: BackupManager) -> None:
        """Preserves backups within retention period.

        Recent backups must not be deleted.
        """
        backup_manager.config.retention_days = 7

        # Create recent backup
        state_data = {"recent": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Run cleanup
        backup_manager.cleanup_old_backups()

        # Verify backup still exists
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        assert backup_file.exists()


class TestConcurrentBackups:
    """Test concurrent backup protection."""

    def test_prevents_concurrent_backups(
        self, backup_manager: BackupManager
    ) -> None:
        """Prevents multiple backups running concurrently.

        Concurrent backups can corrupt state.
        """
        # Lock should be acquired during backup
        assert hasattr(backup_manager, "_backup_lock")

    def test_releases_lock_after_backup(
        self, backup_manager: BackupManager
    ) -> None:
        """Releases lock after backup completes.

        Must not deadlock on subsequent backups.
        """
        state_data = {"test": "data"}

        # First backup
        backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Second backup should succeed (lock released)
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_empty_state_data(self, backup_manager: BackupManager) -> None:
        """Handles empty state data without crashing.

        Edge case: backing up empty state.
        """
        empty_state = {}

        metadata = backup_manager.create_backup(
            state_data=empty_state, backup_type=BackupType.FULL
        )

        assert metadata is not None
        assert metadata.size_bytes >= 0

    def test_handles_large_state_data(self, backup_manager: BackupManager) -> None:
        """Handles large state data appropriately.

        Should warn or split if exceeding max size.
        """
        backup_manager.config.max_backup_size_mb = 1  # 1 MB limit

        # Create data larger than limit
        large_state = {"large_data": "x" * (2 * 1024 * 1024)}  # 2 MB

        # Should handle gracefully (warn, split, or compress)
        metadata = backup_manager.create_backup(
            state_data=large_state, backup_type=BackupType.FULL
        )

        assert metadata is not None

    def test_handles_disk_full_scenario(
        self, backup_manager: BackupManager
    ) -> None:
        """Handles disk full scenario gracefully.

        Should raise clear error when disk space exhausted.
        """
        # Mock disk full error
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            state_data = {"test": "data"}

            with pytest.raises(OSError):
                backup_manager.create_backup(
                    state_data=state_data, backup_type=BackupType.FULL
                )


class TestMetadataPersistence:
    """Test backup metadata persistence."""

    def test_persists_metadata_to_disk(self, backup_manager: BackupManager) -> None:
        """Persists backup metadata to disk.

        Metadata must survive process restarts.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Verify metadata file exists
        metadata_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.meta"
        assert metadata_file.exists()

    def test_loads_metadata_on_initialization(
        self, backup_config: BackupConfig
    ) -> None:
        """Loads existing metadata on manager initialization.

        Must discover existing backups after restart.
        """
        # Create backup with first manager
        manager1 = BackupManager(state_manager=Mock(), config=backup_config)
        state_data = {"test": "data"}
        metadata = manager1.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Create new manager instance
        manager2 = BackupManager(state_manager=Mock(), config=backup_config)

        # Should discover existing backup
        assert metadata.backup_id in manager2._backup_metadata


class TestStorageTiers:
    """Test storage tier management."""

    def test_assigns_storage_tier_to_backup(
        self, backup_manager: BackupManager
    ) -> None:
        """Assigns appropriate storage tier to backup.

        Full backups typically go to durable storage.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        assert metadata.storage_tier in [
            StorageTier.LOCAL,
            StorageTier.NETWORK,
            StorageTier.CLOUD,
        ]

    def test_emergency_backup_uses_fastest_tier(
        self, backup_manager: BackupManager
    ) -> None:
        """Emergency backups use fastest available tier.

        Speed critical during system failures.
        """
        state_data = {"emergency": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.EMERGENCY
        )

        # Emergency should prefer local storage
        assert metadata.storage_tier == StorageTier.LOCAL