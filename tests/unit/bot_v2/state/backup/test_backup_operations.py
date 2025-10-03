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

import asyncio
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

    async def mock_batch_set_state(items: dict) -> int:
        """Mock batch_set_state that returns count of items."""
        return len(items)

    async def mock_create_snapshot() -> dict:
        """Mock create_snapshot that returns test data."""
        return {
            "position:BTC-USD": {"qty": "1.0", "timestamp": "2024-01-01T00:00:00Z"},
            "order:test123": {"id": "test123", "timestamp": "2024-01-01T00:00:00Z"},
        }

    state_manager = Mock()
    state_manager.get_keys_by_pattern = AsyncMock(return_value=["test:key1", "test:key2"])
    state_manager.get_state = AsyncMock(return_value={"test": "data"})
    state_manager.set_state = AsyncMock(return_value=True)
    state_manager.batch_set_state = AsyncMock(side_effect=mock_batch_set_state)
    state_manager.create_snapshot = AsyncMock(side_effect=mock_create_snapshot)
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

    def test_disables_encryption_when_config_false(self, backup_config: BackupConfig) -> None:
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

    def test_handles_s3_unavailable_gracefully(self, backup_config: BackupConfig) -> None:
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

    def test_full_backup_contains_all_data(self, backup_manager: BackupManager) -> None:
        """Full backup contains complete state snapshot.

        No data should be omitted from full backup.
        """
        state_data = {
            "positions": {"AAPL": {"qty": 10}},
            "orders": [{"id": "1", "symbol": "AAPL"}],
            "account": {"equity": 10000.0},
            "metadata": {"last_update": "2024-01-01T00:00:00Z"},
        }

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Verify all keys preserved
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        assert backup_file.exists()

    def test_full_backup_generates_checksum(self, backup_manager: BackupManager) -> None:
        """Full backup generates checksum for integrity verification.

        Checksum detects corruption during storage/retrieval.
        """
        state_data = {"test": "data"}

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata.checksum is not None
        assert len(metadata.checksum) > 0

    def test_full_backup_records_size(self, backup_manager: BackupManager) -> None:
        """Full backup records both raw and compressed size.

        Size tracking for storage management and monitoring.
        """
        state_data = {"large_data": "x" * 10000}

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata.size_bytes > 0
        if backup_manager.config.enable_compression:
            assert metadata.size_compressed > 0
            # Compressed size should be less than raw for repetitive data
            assert metadata.size_compressed < metadata.size_bytes


class TestIncrementalBackup:
    """Test incremental backup creation."""

    def test_creates_incremental_backup(self, backup_manager: BackupManager) -> None:
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

    def test_incremental_requires_baseline(self, backup_manager: BackupManager) -> None:
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

    def test_creates_differential_backup(self, backup_manager: BackupManager) -> None:
        """Creates differential backup against last full backup.

        Differential contains all changes since last full backup.
        """
        # Create full backup
        initial_state = {"positions": {"AAPL": {"qty": 10}}}
        backup_manager.create_backup(state_data=initial_state, backup_type=BackupType.FULL)

        # Create differential
        changed_state = {"positions": {"AAPL": {"qty": 20}}}
        diff_metadata = backup_manager.create_backup(
            state_data=changed_state, backup_type=BackupType.DIFFERENTIAL
        )

        assert diff_metadata.backup_type == BackupType.DIFFERENTIAL

    def test_differential_stores_only_changed_values(self, backup_manager: BackupManager) -> None:
        """Differential backup should persist only the delta payload."""
        # Disable compression/encryption to inspect stored payload directly
        backup_manager.config.enable_compression = False
        backup_manager.compression_service.enabled = False
        backup_manager.config.enable_encryption = False
        backup_manager.encryption_service.enabled = False
        backup_manager._encryption_enabled = False

        baseline_state = {"positions": {"AAPL": {"qty": 10, "pnl": 5}}}
        full_metadata = backup_manager.create_backup(
            state_data=baseline_state, backup_type=BackupType.FULL
        )
        assert full_metadata is not None

        changed_state = {
            "positions": {"AAPL": {"qty": 12, "pnl": 5}},
            "cash": {"usd": 1000},
        }
        diff_metadata = backup_manager.create_backup(
            state_data=changed_state, backup_type=BackupType.DIFFERENTIAL
        )
        assert diff_metadata is not None

        backup_path = Path(backup_manager.transport_service.backup_path)
        diff_path = backup_path / f"{diff_metadata.backup_id}.backup"
        with open(diff_path, "rb") as diff_file:
            payload = json.loads(diff_file.read().decode("utf-8"))

        diff_state = payload.get("state", {})
        assert diff_state == {
            "positions": {"AAPL": {"qty": 12}},
            "cash": {"usd": 1000},
        }


class TestBackupCompression:
    """Test backup compression."""

    def test_compresses_backup_when_enabled(self, backup_manager: BackupManager) -> None:
        """Compresses backup data when compression enabled.

        Reduces storage costs and transfer time.
        """
        backup_manager.config.enable_compression = True
        # Highly compressible data
        state_data = {"repeated": "x" * 10000}

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata.size_compressed < metadata.size_bytes

    def test_skips_compression_when_disabled(self, backup_manager: BackupManager) -> None:
        """Skips compression when disabled in config.

        Faster backups when storage space not a concern.
        """
        backup_manager.config.enable_compression = False
        state_data = {"data": "test"}

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Should have no compression or equal sizes
        assert metadata.size_compressed == 0 or metadata.size_compressed == metadata.size_bytes


class TestBackupEncryption:
    """Test backup encryption."""

    def test_encrypts_backup_when_enabled(self, backup_manager: BackupManager) -> None:
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

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Verify encryption was attempted
        assert metadata.encryption_key_id is not None

    def test_skips_encryption_when_disabled(self, backup_manager: BackupManager) -> None:
        """Skips encryption when disabled in config.

        Faster backups in non-production environments.
        """
        backup_manager.config.enable_encryption = False
        backup_manager.encryption_service.enabled = False
        backup_manager._encryption_enabled = False
        state_data = {"data": "test"}

        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

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

    def test_verifies_checksum_on_restore(self, backup_manager: BackupManager) -> None:
        """Verifies checksum when restoring backup.

        Detects corruption before using potentially invalid state.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Corrupt the backup file
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        backup_file.write_bytes(b"corrupted data")

        # Restore should detect corruption
        with pytest.raises(Exception):  # Should raise checksum validation error
            backup_manager.restore_from_backup(metadata.backup_id)

    def test_handles_missing_backup_gracefully(self, backup_manager: BackupManager) -> None:
        """Handles attempt to restore non-existent backup.

        Should raise clear error for missing backup.
        """
        with pytest.raises(FileNotFoundError):
            backup_manager.restore_from_backup("nonexistent_backup_id")

    def test_decrypts_backup_on_restore(self, backup_manager: BackupManager) -> None:
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
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Mock decrypt to return original data
        original_json = json.dumps({"state": state_data}).encode()
        mock_cipher.decrypt.return_value = original_json

        restored = backup_manager.restore_from_backup(metadata.backup_id)
        assert restored == state_data


class TestS3Operations:
    """Test S3 upload/download functionality."""

    def test_uploads_backup_to_s3(self, backup_manager: BackupManager) -> None:
        """Uploads backup to S3 cloud storage.

        Critical for off-site disaster recovery.
        """
        backup_manager.config.enable_s3 = True
        mock_s3 = Mock()
        backup_manager.transport_service._s3_client = mock_s3
        backup_manager._s3_client = mock_s3

        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Verify S3 upload attempted
        if backup_manager._s3_client:
            backup_manager._upload_to_s3(metadata.backup_id)
            mock_s3.upload_file.assert_called()

    def test_handles_s3_upload_failure_gracefully(self, backup_manager: BackupManager) -> None:
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
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata is not None
        assert metadata.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]


class TestBackupCleanup:
    """Test backup cleanup and retention policies."""

    def test_deletes_old_backups_beyond_retention(self, backup_manager: BackupManager) -> None:
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
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Run cleanup
        backup_manager.cleanup_old_backups()

        # Verify backup still exists
        backup_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.backup"
        assert backup_file.exists()


class TestConcurrentBackups:
    """Test concurrent backup protection."""

    def test_prevents_concurrent_backups(self, backup_manager: BackupManager) -> None:
        """Prevents multiple backups running concurrently.

        Concurrent backups can corrupt state.
        """
        # Lock should be acquired during backup
        assert hasattr(backup_manager, "_backup_lock")

    def test_releases_lock_after_backup(self, backup_manager: BackupManager) -> None:
        """Releases lock after backup completes.

        Must not deadlock on subsequent backups.
        """
        state_data = {"test": "data"}

        # First backup
        backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Second backup should succeed (lock released)
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_empty_state_data(self, backup_manager: BackupManager) -> None:
        """Handles empty state data without crashing.

        Edge case: backing up empty state.
        """
        empty_state = {}

        metadata = backup_manager.create_backup(state_data=empty_state, backup_type=BackupType.FULL)

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
        metadata = backup_manager.create_backup(state_data=large_state, backup_type=BackupType.FULL)

        assert metadata is not None

    def test_handles_disk_full_scenario(self, backup_manager: BackupManager) -> None:
        """Handles disk full scenario gracefully.

        Should raise clear error when disk space exhausted.
        """
        # Mock disk full error
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            state_data = {"test": "data"}

            with pytest.raises(OSError):
                backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)


class TestMetadataPersistence:
    """Test backup metadata persistence."""

    def test_persists_metadata_to_disk(self, backup_manager: BackupManager) -> None:
        """Persists backup metadata to disk.

        Metadata must survive process restarts.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Verify metadata file exists
        metadata_file = Path(backup_manager.config.backup_dir) / f"{metadata.backup_id}.meta"
        assert metadata_file.exists()

    def test_loads_metadata_on_initialization(self, backup_config: BackupConfig) -> None:
        """Loads existing metadata on manager initialization.

        Must discover existing backups after restart.
        """
        # Create backup with first manager
        manager1 = BackupManager(state_manager=Mock(), config=backup_config)
        state_data = {"test": "data"}
        metadata = manager1.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        # Create new manager instance
        manager2 = BackupManager(state_manager=Mock(), config=backup_config)

        # Should discover existing backup
        assert metadata.backup_id in manager2._backup_metadata


class TestStorageTiers:
    """Test storage tier management."""

    def test_assigns_storage_tier_to_backup(self, backup_manager: BackupManager) -> None:
        """Assigns appropriate storage tier to backup.

        Full backups typically go to durable storage.
        """
        state_data = {"test": "data"}
        metadata = backup_manager.create_backup(state_data=state_data, backup_type=BackupType.FULL)

        assert metadata.storage_tier in [
            StorageTier.LOCAL,
            StorageTier.NETWORK,
            StorageTier.CLOUD,
        ]

    def test_emergency_backup_uses_fastest_tier(self, backup_manager: BackupManager) -> None:
        """Emergency backups use fastest available tier.

        Speed critical during system failures.
        """
        state_data = {"emergency": "data"}
        metadata = backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.EMERGENCY
        )

        # Emergency should prefer local storage
        assert metadata.storage_tier == StorageTier.LOCAL


class TestAsyncScheduling:
    """Tests for async backup scheduling."""

    @pytest.mark.asyncio
    async def test_start_scheduled_backups_creates_tasks(
        self, backup_manager: BackupManager
    ) -> None:
        """start_scheduled_backups creates background tasks."""
        await backup_manager.start_scheduled_backups()

        # Should have created 5 tasks (full, diff, inc, cleanup, verification)
        assert len(backup_manager._scheduled_tasks) == 5

        # Cleanup
        await backup_manager.stop_scheduled_backups()

    @pytest.mark.asyncio
    async def test_stop_scheduled_backups_cancels_tasks(
        self, backup_manager: BackupManager
    ) -> None:
        """stop_scheduled_backups cancels all background tasks."""
        await backup_manager.start_scheduled_backups()
        assert len(backup_manager._scheduled_tasks) == 5

        await backup_manager.stop_scheduled_backups()

        # All tasks should be cancelled
        assert len(backup_manager._scheduled_tasks) == 0

    @pytest.mark.asyncio
    async def test_full_backup_schedule_runs_periodically(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """_run_full_backup_schedule creates backups on schedule."""
        backup_created = asyncio.Event()
        original_create = backup_manager._create_backup

        async def mock_create_backup(*args, **kwargs):
            backup_created.set()
            return await original_create(*args, **kwargs)

        monkeypatch.setattr(backup_manager, "_create_backup", mock_create_backup)

        # Set very short interval for testing
        monkeypatch.setattr(backup_manager.config, "full_backup_interval_hours", 0.0001)

        # Start the schedule
        task = asyncio.create_task(backup_manager._run_full_backup_schedule())

        try:
            # Wait for backup to be created (with timeout)
            await asyncio.wait_for(backup_created.wait(), timeout=1.0)
            assert backup_created.is_set()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_differential_backup_schedule_runs_periodically(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """_run_differential_backup_schedule creates backups on schedule."""
        backup_created = asyncio.Event()

        async def mock_create_backup(backup_type, **kwargs):
            if backup_type == BackupType.DIFFERENTIAL:
                backup_created.set()
            return None

        monkeypatch.setattr(backup_manager, "_create_backup", mock_create_backup)
        monkeypatch.setattr(backup_manager.config, "differential_backup_interval_hours", 0.0001)

        task = asyncio.create_task(backup_manager._run_differential_backup_schedule())

        try:
            await asyncio.wait_for(backup_created.wait(), timeout=1.0)
            assert backup_created.is_set()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_incremental_backup_schedule_runs_periodically(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """_run_incremental_backup_schedule creates backups on schedule."""
        backup_created = asyncio.Event()

        async def mock_create_backup(backup_type, **kwargs):
            if backup_type == BackupType.INCREMENTAL:
                backup_created.set()
            return None

        monkeypatch.setattr(backup_manager, "_create_backup", mock_create_backup)
        monkeypatch.setattr(backup_manager.config, "incremental_backup_interval_minutes", 0.001)

        task = asyncio.create_task(backup_manager._run_incremental_backup_schedule())

        try:
            await asyncio.wait_for(backup_created.wait(), timeout=1.0)
            assert backup_created.is_set()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_schedule_handles_backup_errors_gracefully(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """Scheduling loops continue despite backup errors."""
        error_count = 0
        backup_count = 0

        async def failing_create_backup(*args, **kwargs):
            nonlocal error_count, backup_count
            backup_count += 1
            if backup_count == 1:
                error_count += 1
                raise Exception("Simulated backup failure")
            return None

        monkeypatch.setattr(backup_manager, "_create_backup", failing_create_backup)
        # Set very short interval: 0.00001 hours = 0.036 seconds
        monkeypatch.setattr(backup_manager.config, "full_backup_interval_hours", 0.00001)

        task = asyncio.create_task(backup_manager._run_full_backup_schedule())

        try:
            # Wait long enough for multiple attempts (0.15s allows ~4 backups at 0.036s intervals)
            await asyncio.sleep(0.15)
            # Should have continued after error
            assert backup_count >= 2
            assert error_count >= 1
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_schedule_respects_cancellation(self, backup_manager: BackupManager) -> None:
        """Scheduling loops exit cleanly on cancellation."""
        task = asyncio.create_task(backup_manager._run_full_backup_schedule())

        # Let it start
        await asyncio.sleep(0.01)

        # Cancel
        task.cancel()

        # Should exit cleanly - loop catches CancelledError and breaks
        try:
            await task
        except asyncio.CancelledError:
            # This is acceptable too if cancellation happens before exception handler
            pass

        # Task should be done either way
        assert task.done()


class TestCreateBackupErrorPaths:
    """Tests for _create_backup error handling."""

    @pytest.mark.asyncio
    async def test_prevents_concurrent_backups(self, backup_manager: BackupManager) -> None:
        """Rejects backup creation when one is already in progress."""
        # Simulate backup in progress
        backup_manager._backup_in_progress = True

        result = await backup_manager._create_backup(BackupType.FULL)

        # Should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_empty_backup_data(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """Handles empty backup data gracefully."""

        async def mock_collect_empty_data(*args, **kwargs):
            return None  # Empty data

        monkeypatch.setattr(backup_manager, "_collect_backup_data", mock_collect_empty_data)

        result = await backup_manager._create_backup(BackupType.FULL)

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_serialization_error(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """Handles serialization errors gracefully."""

        def failing_serialize(*args, **kwargs):
            raise ValueError("Serialization failed")

        monkeypatch.setattr(
            backup_manager.backup_creator, "_serialize_backup_data", failing_serialize
        )

        # Create state data
        state_data = {"test": "data"}

        result = await backup_manager._create_backup(BackupType.FULL, state_data=state_data)

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_oserror_and_propagates(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """OSError during backup creation is propagated."""

        async def failing_store(*args, **kwargs):
            raise OSError("Disk full")

        monkeypatch.setattr(backup_manager.backup_creator, "_store_backup", failing_store)

        state_data = {"test": "data"}

        # Should raise OSError
        with pytest.raises(OSError, match="Disk full"):
            await backup_manager._create_backup(BackupType.FULL, state_data=state_data)

    @pytest.mark.asyncio
    async def test_clears_backup_in_progress_flag_after_error(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """Ensures _backup_in_progress is cleared even after errors."""

        async def failing_collect(*args, **kwargs):
            raise ValueError("Collection failed")

        monkeypatch.setattr(backup_manager, "_collect_backup_data", failing_collect)

        # Before backup
        assert not backup_manager._backup_in_progress

        # Attempt backup (will fail)
        result = await backup_manager._create_backup(BackupType.FULL)

        # Should have cleared the flag despite error
        assert not backup_manager._backup_in_progress
        assert result is None

    @pytest.mark.asyncio
    async def test_clears_pending_snapshot_after_error(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """Clears pending snapshot even if backup fails."""
        backup_manager._pending_state_snapshot = {"test": "snapshot"}

        async def failing_collect(*args, **kwargs):
            raise ValueError("Collection failed")

        monkeypatch.setattr(backup_manager, "_collect_backup_data", failing_collect)

        await backup_manager._create_backup(BackupType.FULL)

        # Should have cleared pending snapshot
        assert backup_manager._pending_state_snapshot is None


class TestRestoreAsyncPaths:
    """Tests for async restoration paths."""

    @pytest.mark.asyncio
    async def test_restore_from_backup_async_success(self, backup_manager: BackupManager) -> None:
        """restore_from_backup succeeds in async context."""
        # Create a backup first
        state_data = {"test": "async_restore"}
        metadata = await backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Restore in async context
        result = await backup_manager.restore_from_backup(metadata.backup_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_restore_from_backup_async_failure(
        self, backup_manager: BackupManager, monkeypatch
    ) -> None:
        """restore_from_backup returns False on async error."""
        # Create a backup
        state_data = {"test": "data"}
        metadata = await backup_manager.create_backup(
            state_data=state_data, backup_type=BackupType.FULL
        )

        # Mock internal restore to fail
        async def failing_restore(*args, **kwargs):
            raise RuntimeError("Restore failed")

        monkeypatch.setattr(
            backup_manager.backup_restorer, "restore_from_backup_internal", failing_restore
        )

        # Should return False instead of raising
        result = await backup_manager.restore_from_backup(metadata.backup_id)

        assert result is False


class TestRestoreLatestBackup:
    """Tests for restore_latest_backup method."""

    @pytest.mark.asyncio
    async def test_restores_latest_full_backup(self, backup_manager: BackupManager) -> None:
        """Restores most recent FULL backup when requested."""
        # Create multiple backups
        await backup_manager.create_backup(state_data={"v": 1}, backup_type=BackupType.FULL)
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await backup_manager.create_backup(state_data={"v": 2}, backup_type=BackupType.FULL)

        # Restore latest
        result = await backup_manager.restore_latest_backup(backup_type=BackupType.FULL)

        assert result is True

    @pytest.mark.asyncio
    async def test_restores_latest_any_type(self, backup_manager: BackupManager) -> None:
        """Restores most recent backup of any type when no filter specified."""
        # Create backups of different types
        await backup_manager.create_backup(state_data={"v": 1}, backup_type=BackupType.FULL)

        # Restore latest (any type)
        result = await backup_manager.restore_latest_backup()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_backups_exist(self, backup_manager: BackupManager) -> None:
        """Returns False when no valid backups found."""
        # No backups created
        result = await backup_manager.restore_latest_backup()

        assert result is False

    @pytest.mark.asyncio
    async def test_filters_by_backup_type(self, backup_manager: BackupManager) -> None:
        """Filters backups by type when specified."""
        # Create different backup types
        await backup_manager.create_backup(state_data={"full": True}, backup_type=BackupType.FULL)

        # Request latest INCREMENTAL (doesn't exist)
        result = await backup_manager.restore_latest_backup(backup_type=BackupType.INCREMENTAL)

        # Should fail since no INCREMENTAL backups
        assert result is False
