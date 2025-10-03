"""
Comprehensive tests for BackupStorage.

Tests cover:
- Initialization and path setup
- Local storage (store/retrieve/delete)
- Network storage (store/retrieve/delete)
- S3 cloud storage (store/retrieve/delete with mocked boto3)
- S3 archive tier with GLACIER storage class
- S3 initialization and fallback behavior
- Error handling and graceful degradation
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from bot_v2.state.backup.storage import BackupStorage
from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupType,
    BackupStatus,
    StorageTier,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_backup_dir(tmp_path):
    """Create temporary backup directory structure."""
    backup_dir = tmp_path / "backups"
    local_dir = tmp_path / "local"
    network_dir = tmp_path / "network"

    backup_dir.mkdir()
    local_dir.mkdir()
    network_dir.mkdir()

    return {
        "backup_dir": str(backup_dir),
        "local_dir": str(local_dir),
        "network_dir": str(network_dir),
    }


@pytest.fixture
def backup_config(temp_backup_dir):
    """Create test backup configuration."""
    return BackupConfig(
        backup_dir=temp_backup_dir["backup_dir"],
        local_storage_path=temp_backup_dir["local_dir"],
        network_storage_path=temp_backup_dir["network_dir"],
        s3_bucket="test-bucket",
        s3_region="us-east-1",
    )


@pytest.fixture
def backup_storage(backup_config):
    """Create BackupStorage instance with test config."""
    with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
        storage = BackupStorage(backup_config)
    return storage


@pytest.fixture
def sample_metadata():
    """Create sample backup metadata."""
    return BackupMetadata(
        backup_id="test-backup-123",
        backup_type=BackupType.FULL,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        size_bytes=1024,
        size_compressed=512,
        checksum="abc123",
        encryption_key_id=None,
        storage_tier=StorageTier.LOCAL,
        retention_days=30,
        status=BackupStatus.COMPLETED,
    )


# ============================================================================
# Test: Initialization
# ============================================================================


class TestBackupStorageInitialization:
    """Test BackupStorage initialization and path setup."""

    def test_initialization_creates_directories(self, temp_backup_dir):
        """Test that initialization creates all required directories."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=temp_backup_dir["network_dir"],
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        assert Path(config.backup_dir).exists()
        assert Path(config.local_storage_path).exists()
        assert Path(config.network_storage_path).exists()

    def test_initialization_without_network_path(self, temp_backup_dir):
        """Test initialization without network storage path."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=None,
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        assert storage.config.network_storage_path is None

    def test_initialization_without_s3(self, backup_config):
        """Test initialization when S3 is not available."""
        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(backup_config)

        assert storage.s3_client is None

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    def test_initialization_with_s3_success(self, mock_boto3, backup_config):
        """Test successful S3 initialization."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        mock_boto3.client.assert_called_once_with("s3", region_name="us-east-1")
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        assert storage.s3_client is mock_client

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    def test_initialization_with_s3_failure(self, mock_boto3, backup_config):
        """Test S3 initialization failure."""
        mock_client = Mock()
        mock_client.head_bucket.side_effect = Exception("Bucket not found")
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        assert storage.s3_client is None


# ============================================================================
# Test: Local Storage Operations
# ============================================================================


class TestBackupStorageLocal:
    """Test local storage operations."""

    @pytest.mark.asyncio
    async def test_store_local(self, backup_storage, sample_metadata):
        """Test storing backup to local filesystem."""
        test_data = b"test backup data"

        result = await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.LOCAL,
        )

        # Verify file was created
        expected_path = Path(backup_storage.config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)

        # Verify content
        with open(expected_path, "rb") as f:
            assert f.read() == test_data

    @pytest.mark.asyncio
    async def test_retrieve_local(self, backup_storage, sample_metadata):
        """Test retrieving backup from local filesystem."""
        test_data = b"test backup data"

        # Store first
        await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.LOCAL,
        )

        # Retrieve
        sample_metadata.storage_tier = StorageTier.LOCAL
        retrieved_data = await backup_storage.retrieve(sample_metadata)

        assert retrieved_data == test_data

    @pytest.mark.asyncio
    async def test_retrieve_local_nonexistent(self, backup_storage, sample_metadata):
        """Test retrieving nonexistent local backup returns None."""
        sample_metadata.storage_tier = StorageTier.LOCAL
        sample_metadata.backup_id = "nonexistent-backup"

        retrieved_data = await backup_storage.retrieve(sample_metadata)

        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_delete_local(self, backup_storage, sample_metadata):
        """Test deleting backup from local filesystem."""
        test_data = b"test backup data"

        # Store first
        await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.LOCAL,
        )

        sample_metadata.storage_tier = StorageTier.LOCAL
        result = await backup_storage.delete(sample_metadata)

        assert result is True

        # Verify file was deleted
        expected_path = Path(backup_storage.config.local_storage_path) / "test-backup-123.backup"
        assert not expected_path.exists()

    @pytest.mark.asyncio
    async def test_delete_local_nonexistent(self, backup_storage, sample_metadata):
        """Test deleting nonexistent local backup still returns True."""
        sample_metadata.storage_tier = StorageTier.LOCAL
        sample_metadata.backup_id = "nonexistent-backup"

        result = await backup_storage.delete(sample_metadata)

        assert result is True


# ============================================================================
# Test: Network Storage Operations
# ============================================================================


class TestBackupStorageNetwork:
    """Test network storage operations."""

    @pytest.mark.asyncio
    async def test_store_network(self, backup_storage, sample_metadata):
        """Test storing backup to network filesystem."""
        test_data = b"test backup data"

        result = await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.NETWORK,
        )

        # Verify file was created
        expected_path = Path(backup_storage.config.network_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)

        # Verify content
        with open(expected_path, "rb") as f:
            assert f.read() == test_data

    @pytest.mark.asyncio
    async def test_store_network_without_path_fallsback_to_local(self, temp_backup_dir):
        """Test storing to network without path configured falls back to local."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=None,
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        test_data = b"test backup data"

        result = await storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.NETWORK,
        )

        # Should fall back to local
        expected_path = Path(config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_retrieve_network(self, backup_storage, sample_metadata):
        """Test retrieving backup from network filesystem."""
        test_data = b"test backup data"

        # Store first
        await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.NETWORK,
        )

        # Retrieve
        sample_metadata.storage_tier = StorageTier.NETWORK
        retrieved_data = await backup_storage.retrieve(sample_metadata)

        assert retrieved_data == test_data

    @pytest.mark.asyncio
    async def test_retrieve_network_nonexistent(self, backup_storage, sample_metadata):
        """Test retrieving nonexistent network backup returns None."""
        sample_metadata.storage_tier = StorageTier.NETWORK
        sample_metadata.backup_id = "nonexistent-backup"

        retrieved_data = await backup_storage.retrieve(sample_metadata)

        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_delete_network(self, backup_storage, sample_metadata):
        """Test deleting backup from network filesystem."""
        test_data = b"test backup data"

        # Store first
        await backup_storage.store(
            backup_id=sample_metadata.backup_id,
            data=test_data,
            tier=StorageTier.NETWORK,
        )

        sample_metadata.storage_tier = StorageTier.NETWORK
        result = await backup_storage.delete(sample_metadata)

        assert result is True

        # Verify file was deleted
        expected_path = Path(backup_storage.config.network_storage_path) / "test-backup-123.backup"
        assert not expected_path.exists()


# ============================================================================
# Test: S3 Cloud Storage Operations
# ============================================================================


class TestBackupStorageS3Cloud:
    """Test S3 cloud storage operations."""

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_store_cloud(self, mock_boto3, backup_config):
        """Test storing backup to S3 cloud tier."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        test_data = b"test backup data"

        result = await storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.CLOUD,
        )

        # Verify S3 put_object was called with correct parameters
        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="backups/test-backup-123.backup",
            Body=test_data,
            StorageClass="STANDARD_IA",
        )

        assert result == "s3://test-bucket/backups/test-backup-123.backup"

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_store_cloud_failure_fallsback_to_local(self, mock_boto3, backup_config):
        """Test S3 store failure falls back to local storage."""
        mock_client = Mock()
        mock_client.put_object.side_effect = Exception("S3 error")
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        test_data = b"test backup data"

        result = await storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.CLOUD,
        )

        # Should fall back to local
        expected_path = Path(backup_config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_retrieve_cloud(self, mock_boto3, backup_config, sample_metadata):
        """Test retrieving backup from S3 cloud tier."""
        mock_client = Mock()
        mock_response = {"Body": Mock()}
        test_data = b"test backup data"
        mock_response["Body"].read.return_value = test_data
        mock_client.get_object.return_value = mock_response
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        sample_metadata.storage_tier = StorageTier.CLOUD

        retrieved_data = await storage.retrieve(sample_metadata)

        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="backups/test-backup-123.backup",
        )

        assert retrieved_data == test_data

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_retrieve_cloud_no_body(self, mock_boto3, backup_config, sample_metadata):
        """Test retrieving from S3 when response has no Body."""
        mock_client = Mock()
        mock_response = {}  # No Body key
        mock_client.get_object.return_value = mock_response
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        sample_metadata.storage_tier = StorageTier.CLOUD

        retrieved_data = await storage.retrieve(sample_metadata)

        assert retrieved_data is None

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_delete_cloud(self, mock_boto3, backup_config, sample_metadata):
        """Test deleting backup from S3 cloud tier."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        sample_metadata.storage_tier = StorageTier.CLOUD

        result = await storage.delete(sample_metadata)

        mock_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="backups/test-backup-123.backup",
        )

        assert result is True


# ============================================================================
# Test: S3 Archive Tier Operations
# ============================================================================


class TestBackupStorageS3Archive:
    """Test S3 archive tier operations with GLACIER storage class."""

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_store_archive(self, mock_boto3, backup_config):
        """Test storing backup to S3 archive tier with GLACIER storage class."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        test_data = b"test backup data"

        result = await storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.ARCHIVE,
        )

        # Verify S3 put_object was called with GLACIER storage class
        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="archive/test-backup-123.backup",
            Body=test_data,
            StorageClass="GLACIER",
        )

        assert result == "s3://test-bucket/archive/test-backup-123.backup"

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_retrieve_archive(self, mock_boto3, backup_config, sample_metadata):
        """Test retrieving backup from S3 archive tier."""
        mock_client = Mock()
        mock_response = {"Body": Mock()}
        test_data = b"test backup data"
        mock_response["Body"].read.return_value = test_data
        mock_client.get_object.return_value = mock_response
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        sample_metadata.storage_tier = StorageTier.ARCHIVE

        retrieved_data = await storage.retrieve(sample_metadata)

        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="archive/test-backup-123.backup",
        )

        assert retrieved_data == test_data

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_delete_archive(self, mock_boto3, backup_config, sample_metadata):
        """Test deleting backup from S3 archive tier."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        storage = BackupStorage(backup_config)

        sample_metadata.storage_tier = StorageTier.ARCHIVE

        result = await storage.delete(sample_metadata)

        mock_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="archive/test-backup-123.backup",
        )

        assert result is True


# ============================================================================
# Test: Fallback Behavior
# ============================================================================


class TestBackupStorageFallback:
    """Test fallback behavior when storage tiers are unavailable."""

    @pytest.mark.asyncio
    async def test_store_cloud_without_s3_fallsback_to_local(self, backup_storage):
        """Test storing to cloud without S3 falls back to local."""
        # backup_storage fixture has S3 disabled
        test_data = b"test backup data"

        result = await backup_storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.CLOUD,
        )

        # Should fall back to local
        expected_path = Path(backup_storage.config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)

    @pytest.mark.asyncio
    async def test_store_archive_without_s3_fallsback_to_local(self, backup_storage):
        """Test storing to archive without S3 falls back to local."""
        # backup_storage fixture has S3 disabled
        test_data = b"test backup data"

        result = await backup_storage.store(
            backup_id="test-backup-123",
            data=test_data,
            tier=StorageTier.ARCHIVE,
        )

        # Should fall back to local
        expected_path = Path(backup_storage.config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)

    @pytest.mark.asyncio
    async def test_retrieve_cloud_without_s3_returns_none(self, backup_storage, sample_metadata):
        """Test retrieving from cloud without S3 returns None."""
        sample_metadata.storage_tier = StorageTier.CLOUD

        retrieved_data = await backup_storage.retrieve(sample_metadata)

        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_delete_cloud_without_s3_returns_false(self, backup_storage, sample_metadata):
        """Test deleting from cloud without S3 returns False."""
        sample_metadata.storage_tier = StorageTier.CLOUD

        result = await backup_storage.delete(sample_metadata)

        assert result is False


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestBackupStorageEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_store_network_raises_error_without_path(self, temp_backup_dir):
        """Test storing to network without configured path."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=None,
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        # Direct call to _store_network should raise ValueError
        with pytest.raises(ValueError, match="Network storage path not configured"):
            storage._store_network("test-backup-123", b"test data")

    @pytest.mark.asyncio
    async def test_retrieve_network_without_path_returns_none(
        self, temp_backup_dir, sample_metadata
    ):
        """Test retrieving from network without configured path returns None."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=None,
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        sample_metadata.storage_tier = StorageTier.NETWORK

        retrieved_data = await storage.retrieve(sample_metadata)

        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_delete_network_without_path_returns_false(
        self, temp_backup_dir, sample_metadata
    ):
        """Test deleting from network without configured path returns False."""
        config = BackupConfig(
            backup_dir=temp_backup_dir["backup_dir"],
            local_storage_path=temp_backup_dir["local_dir"],
            network_storage_path=None,
        )

        with patch("bot_v2.state.backup.storage.S3_AVAILABLE", False):
            storage = BackupStorage(config)

        sample_metadata.storage_tier = StorageTier.NETWORK

        result = await storage.delete(sample_metadata)

        assert result is False

    @patch("bot_v2.state.backup.storage.boto3")
    @patch("bot_v2.state.backup.storage.S3_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_store_s3_without_client_fallsback_to_local(self, mock_boto3, backup_config):
        """Test storing to S3 when client is None falls back to local."""
        mock_boto3.client.return_value = Mock()

        storage = BackupStorage(backup_config)
        storage.s3_client = None  # Simulate S3 client unavailable

        test_data = b"test backup data"

        result = await storage._store_s3(
            backup_id="test-backup-123",
            data=test_data,
            prefix="backups",
        )

        # Should fall back to local
        expected_path = Path(backup_config.local_storage_path) / "test-backup-123.backup"
        assert expected_path.exists()
        assert result == str(expected_path)
