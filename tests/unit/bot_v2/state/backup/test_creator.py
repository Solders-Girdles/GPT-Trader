"""Unit tests for BackupCreator.

Tests the backup creation pipeline in isolation:
- Serialization and compression
- Encryption and metadata generation
- Storage tier selection
- Verification and baseline updates
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.backup.creator import BackupCreator
from bot_v2.state.backup.models import (
    BackupConfig,
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)


@pytest.fixture
def backup_config():
    """Minimal backup configuration."""
    return BackupConfig(
        backup_dir="/tmp/test_backups",
        enable_encryption=True,
        enable_compression=True,
        verify_after_backup=False,  # Disable for most tests
    )


@pytest.fixture
def backup_context():
    """Shared backup context."""
    return BackupContext()


@pytest.fixture
def mock_metadata_manager():
    """Mock metadata manager."""
    manager = Mock()
    manager.add_to_history = Mock()
    return manager


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service."""
    service = Mock()
    service.encrypt = Mock(return_value=(b"encrypted_data", "key_123"))
    service.decrypt = Mock(return_value=b"decrypted_data")
    return service


@pytest.fixture
def mock_compression_service():
    """Mock compression service."""
    service = Mock()
    service.compress = Mock(return_value=(b"compressed", 100, 50))
    service.decompress = Mock(return_value=b"decompressed")
    return service


@pytest.fixture
def mock_transport_service():
    """Mock transport service."""
    service = Mock()
    service.store = AsyncMock(return_value="backup_123")
    service.retrieve = AsyncMock(return_value=b"backup_data")
    return service


@pytest.fixture
def mock_tier_strategy():
    """Mock tier strategy."""
    strategy = Mock()
    strategy.determine_tier = Mock(return_value=StorageTier.LOCAL)
    return strategy


@pytest.fixture
def backup_creator(
    backup_config,
    backup_context,
    mock_metadata_manager,
    mock_encryption_service,
    mock_compression_service,
    mock_transport_service,
    mock_tier_strategy,
):
    """BackupCreator instance with mocked dependencies."""
    return BackupCreator(
        config=backup_config,
        context=backup_context,
        metadata_manager=mock_metadata_manager,
        encryption_service=mock_encryption_service,
        compression_service=mock_compression_service,
        transport_service=mock_transport_service,
        tier_strategy=mock_tier_strategy,
    )


class TestBackupCreation:
    """Tests for backup creation workflow."""

    @pytest.mark.asyncio
    async def test_creates_full_backup_successfully(self, backup_creator):
        """Creates full backup with complete pipeline."""
        backup_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": {"position:BTC": {"qty": 1.5}, "portfolio_current": {"cash": 10000}},
        }
        backup_id = "FULL_20250101_120000"
        start_time = datetime.now(timezone.utc)

        metadata = await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id=backup_id,
            start_time=start_time,
            pending_snapshot=backup_data["state"],
        )

        assert metadata is not None
        assert metadata.backup_id == backup_id
        assert metadata.backup_type == BackupType.FULL
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.checksum is not None
        assert metadata.encryption_key_id == "key_123"

    @pytest.mark.asyncio
    async def test_handles_empty_backup_data(self, backup_creator):
        """Raises exception for empty backup data."""
        with pytest.raises(Exception, match="No data to backup"):
            await backup_creator.create_backup_internal(
                backup_type=BackupType.FULL,
                backup_data={},
                backup_id="test_id",
                start_time=datetime.now(timezone.utc),
                pending_snapshot=None,
            )

    @pytest.mark.asyncio
    async def test_compression_pipeline(self, backup_creator, mock_compression_service):
        """Compresses backup data correctly."""
        backup_data = {"state": {"key": "value"}}
        backup_id = "test_compression"

        await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id=backup_id,
            start_time=datetime.now(timezone.utc),
            pending_snapshot=None,
        )

        # Verify compression was called
        assert mock_compression_service.compress.call_count >= 1

    @pytest.mark.asyncio
    async def test_encryption_pipeline(self, backup_creator, mock_encryption_service):
        """Encrypts backup data correctly."""
        backup_data = {"state": {"key": "value"}}
        backup_id = "test_encryption"

        metadata = await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id=backup_id,
            start_time=datetime.now(timezone.utc),
            pending_snapshot=None,
        )

        # Verify encryption was called
        mock_encryption_service.encrypt.assert_called_once()
        assert metadata.encryption_key_id == "key_123"

    @pytest.mark.asyncio
    async def test_storage_tier_selection(self, backup_creator, mock_tier_strategy):
        """Selects appropriate storage tier for backup type."""
        backup_data = {"state": {"key": "value"}}

        await backup_creator.create_backup_internal(
            backup_type=BackupType.EMERGENCY,
            backup_data=backup_data,
            backup_id="emergency_backup",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=None,
        )

        # Verify tier strategy was consulted
        mock_tier_strategy.determine_tier.assert_called_once_with(BackupType.EMERGENCY)

    @pytest.mark.asyncio
    async def test_updates_baseline_on_full_backup(self, backup_creator, backup_context):
        """Updates baseline snapshots for full backups."""
        snapshot = {"position:BTC": {"qty": 1.5}}
        backup_data = {"state": snapshot}

        await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id="full_backup",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=snapshot,
        )

        # Verify baseline was updated
        assert backup_context.last_full_state == snapshot
        assert backup_context.last_backup_state == snapshot

    @pytest.mark.asyncio
    async def test_updates_baseline_on_snapshot_backup(self, backup_creator, backup_context):
        """Updates baseline snapshots for snapshot backups."""
        snapshot = {"position:ETH": {"qty": 10}}
        backup_data = {"state": snapshot}

        await backup_creator.create_backup_internal(
            backup_type=BackupType.SNAPSHOT,
            backup_data=backup_data,
            backup_id="snapshot_backup",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=snapshot,
        )

        # Verify baseline was updated
        assert backup_context.last_full_state == snapshot
        assert backup_context.last_backup_state == snapshot

    @pytest.mark.asyncio
    async def test_does_not_update_full_baseline_on_incremental(
        self, backup_creator, backup_context
    ):
        """Does not update full baseline for incremental backups."""
        snapshot = {"position:BTC": {"qty": 2.0}}
        backup_data = {"state": snapshot}

        await backup_creator.create_backup_internal(
            backup_type=BackupType.INCREMENTAL,
            backup_data=backup_data,
            backup_id="incremental_backup",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=snapshot,
        )

        # Verify only last_backup_state was updated
        assert backup_context.last_backup_state == snapshot
        assert backup_context.last_full_state is None  # Not updated


class TestBackupVerification:
    """Tests for backup verification."""

    @pytest.mark.asyncio
    async def test_verification_success(self, backup_creator, backup_config):
        """Marks backup as VERIFIED when verification passes."""
        backup_config.verify_after_backup = True
        backup_data = {"state": {"key": "value"}}

        with patch.object(backup_creator, "_verify_backup", return_value=True):
            metadata = await backup_creator.create_backup_internal(
                backup_type=BackupType.FULL,
                backup_data=backup_data,
                backup_id="verified_backup",
                start_time=datetime.now(timezone.utc),
                pending_snapshot=None,
            )

        assert metadata.status == BackupStatus.VERIFIED
        assert metadata.verification_status == "passed"

    @pytest.mark.asyncio
    async def test_verification_failure(self, backup_creator, backup_config):
        """Marks backup as CORRUPTED when verification fails."""
        backup_config.verify_after_backup = True
        backup_data = {"state": {"key": "value"}}

        with patch.object(backup_creator, "_verify_backup", return_value=False):
            metadata = await backup_creator.create_backup_internal(
                backup_type=BackupType.FULL,
                backup_data=backup_data,
                backup_id="failed_verification",
                start_time=datetime.now(timezone.utc),
                pending_snapshot=None,
            )

        assert metadata.status == BackupStatus.CORRUPTED
        assert metadata.verification_status == "failed"


class TestMetadataGeneration:
    """Tests for backup metadata generation."""

    @pytest.mark.asyncio
    async def test_metadata_includes_data_sources(self, backup_creator):
        """Includes data source keys in metadata."""
        backup_data = {"state": {"position:BTC": {}, "order:123": {}, "portfolio_current": {}}}

        metadata = await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id="metadata_test",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=None,
        )

        assert "position:BTC" in metadata.data_sources
        assert "order:123" in metadata.data_sources
        assert "portfolio_current" in metadata.data_sources

    @pytest.mark.asyncio
    async def test_metadata_includes_timing(self, backup_creator):
        """Records backup duration in metadata."""
        backup_data = {"state": {"key": "value"}}
        start_time = datetime.now(timezone.utc)

        metadata = await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id="timing_test",
            start_time=start_time,
            pending_snapshot=None,
        )

        assert metadata.backup_duration_seconds >= 0
        # Timestamp may have timezone stripped by BackupCreator
        assert metadata.timestamp.replace(tzinfo=timezone.utc) == start_time

    @pytest.mark.asyncio
    async def test_metadata_includes_compression_ratio(self, backup_creator):
        """Records compression ratio in metadata."""
        backup_data = {"state": {"key": "value"}}

        metadata = await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id="compression_test",
            start_time=datetime.now(timezone.utc),
            pending_snapshot=None,
        )

        # Verify compression information is present
        assert metadata.size_bytes > 0
        assert metadata.size_compressed == 50  # From mock compression service


class TestHistoryManagement:
    """Tests for backup history management."""

    @pytest.mark.asyncio
    async def test_adds_metadata_to_history(self, backup_creator, mock_metadata_manager):
        """Adds completed backup to history."""
        backup_data = {"state": {"key": "value"}}
        start_time = datetime.now(timezone.utc)

        await backup_creator.create_backup_internal(
            backup_type=BackupType.FULL,
            backup_data=backup_data,
            backup_id="history_test",
            start_time=start_time,
            pending_snapshot=None,
        )

        # Verify metadata was added to history
        mock_metadata_manager.add_to_history.assert_called_once()
        call_args = mock_metadata_manager.add_to_history.call_args[0]
        metadata = call_args[0]
        assert isinstance(metadata, BackupMetadata)
        assert metadata.backup_id == "history_test"
