"""Unit tests for TransportService multi-tier backup storage.

Tests cover:
- Path initialization and error handling
- S3 client initialization and fallback
- Store operations across LOCAL/NETWORK/CLOUD/ARCHIVE tiers
- Retrieve operations with fallback logic
- Delete operations and error handling
- Upload to S3 with error paths
- Storage availability properties
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from bot_v2.state.backup.models import StorageTier
from bot_v2.state.backup.services.transport import TransportService, S3_AVAILABLE

# Skip S3 tests if boto3 not available
skip_if_no_boto3 = pytest.mark.skipif(not S3_AVAILABLE, reason="boto3 not available")


@pytest.fixture
def temp_transport_paths(tmp_path: Path):
    """Create temporary paths for transport testing."""
    paths = {
        "local": tmp_path / "local",
        "backup": tmp_path / "backup",
        "network": tmp_path / "network",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture
def transport_service(temp_transport_paths: dict) -> TransportService:
    """Create TransportService with local paths only (no S3)."""
    return TransportService(
        local_path=temp_transport_paths["local"],
        backup_path=temp_transport_paths["backup"],
        network_path=temp_transport_paths["network"],
        s3_bucket=None,
        enable_s3=False,
    )


class TestPathInitialization:
    """Tests for storage path initialization."""

    def test_init_creates_local_and_backup_paths(self, temp_transport_paths: dict) -> None:
        """Initializes local and backup directories on creation."""
        local_path = temp_transport_paths["local"] / "new_local"
        backup_path = temp_transport_paths["backup"] / "new_backup"

        # Paths don't exist yet
        assert not local_path.exists()
        assert not backup_path.exists()

        service = TransportService(local_path=local_path, backup_path=backup_path, enable_s3=False)

        # Paths should be created
        assert local_path.exists()
        assert backup_path.exists()

    def test_network_path_creation_failure_handled(self, temp_transport_paths: dict) -> None:
        """Handles network path creation failure gracefully."""
        # Use invalid path (like /proc/invalid)
        invalid_network_path = Path("/invalid/network/path")

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            network_path=invalid_network_path,
            enable_s3=False,
        )

        # Network path should be disabled due to error
        assert service.network_path is None

    def test_network_path_none_when_not_provided(self, temp_transport_paths: dict) -> None:
        """Network path is None when not provided."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            network_path=None,
            enable_s3=False,
        )

        assert service.network_path is None


class TestS3Initialization:
    """Tests for S3 client initialization."""

    def test_s3_disabled_when_boto3_unavailable(
        self, temp_transport_paths: dict, monkeypatch
    ) -> None:
        """S3 client is None when boto3 is unavailable."""
        import bot_v2.state.backup.services.transport as transport_module

        monkeypatch.setattr(transport_module, "S3_AVAILABLE", False)
        monkeypatch.setattr(transport_module, "boto3", None)

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        assert service._s3_client is None

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_s3_initialized_when_boto3_available(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """S3 client initialized when boto3 available and bucket valid."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        assert service._s3_client is not None
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_s3_init_failure_sets_client_to_none(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """S3 client is None when initialization fails."""
        mock_s3 = Mock()
        mock_s3.head_bucket.side_effect = Exception("Bucket not found")
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="invalid-bucket",
            enable_s3=True,
        )

        assert service._s3_client is None

    def test_s3_not_initialized_when_disabled(self, temp_transport_paths: dict) -> None:
        """S3 client not initialized when enable_s3=False."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=False,
        )

        assert service._s3_client is None


class TestStoreOperations:
    """Tests for backup storage operations."""

    @pytest.mark.asyncio
    async def test_store_local_tier(self, transport_service: TransportService) -> None:
        """Stores backup to LOCAL tier."""
        backup_id = "TEST_BACKUP_001"
        data = b"test backup data"

        path = await transport_service.store(backup_id, data, StorageTier.LOCAL)

        assert path is not None
        assert Path(path).exists()
        assert Path(path).read_bytes() == data

    @pytest.mark.asyncio
    async def test_store_network_tier(self, transport_service: TransportService) -> None:
        """Stores backup to NETWORK tier."""
        backup_id = "TEST_BACKUP_002"
        data = b"network backup data"

        path = await transport_service.store(backup_id, data, StorageTier.NETWORK)

        assert path is not None
        assert Path(path).exists()
        assert Path(path).read_bytes() == data

    @pytest.mark.asyncio
    async def test_store_network_fallback_to_local_when_unavailable(
        self, temp_transport_paths: dict
    ) -> None:
        """Falls back to LOCAL when NETWORK storage unavailable."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            network_path=None,  # Network not available
            enable_s3=False,
        )

        backup_id = "TEST_BACKUP_003"
        data = b"fallback data"

        path = await service.store(backup_id, data, StorageTier.NETWORK)

        # Should fall back to backup_path
        assert str(temp_transport_paths["backup"]) in path

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_store_cloud_tier_to_s3(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Stores backup to CLOUD tier (S3)."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.put_object = Mock(return_value={})
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        backup_id = "TEST_BACKUP_004"
        data = b"cloud backup data"

        path = await service.store(backup_id, data, StorageTier.CLOUD)

        assert path.startswith("s3://test-bucket/")
        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args
        assert call_args.kwargs["Bucket"] == "test-bucket"
        assert call_args.kwargs["Body"] == data
        assert call_args.kwargs["StorageClass"] == "STANDARD_IA"

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_store_archive_tier_to_glacier(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Stores backup to ARCHIVE tier (Glacier)."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.put_object = Mock(return_value={})
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        backup_id = "TEST_BACKUP_005"
        data = b"archive backup data"

        path = await service.store(backup_id, data, StorageTier.ARCHIVE)

        assert path.startswith("s3://test-bucket/archive/")
        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args
        assert call_args.kwargs["StorageClass"] == "GLACIER"

    @pytest.mark.asyncio
    async def test_store_cloud_fallback_to_local_when_s3_unavailable(
        self, temp_transport_paths: dict
    ) -> None:
        """Falls back to LOCAL when CLOUD storage unavailable."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket=None,  # S3 not available
            enable_s3=False,
        )

        backup_id = "TEST_BACKUP_006"
        data = b"fallback cloud data"

        path = await service.store(backup_id, data, StorageTier.CLOUD)

        # Should fall back to local
        assert str(temp_transport_paths["backup"]) in path

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_store_s3_error_falls_back_to_local(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Falls back to LOCAL when S3 put_object fails."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.put_object.side_effect = Exception("S3 error")
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        backup_id = "TEST_BACKUP_007"
        data = b"s3 error data"

        path = await service.store(backup_id, data, StorageTier.CLOUD)

        # Should fall back to local
        assert str(temp_transport_paths["backup"]) in path
        # Verify data was saved locally
        assert Path(path).read_bytes() == data

    @pytest.mark.asyncio
    async def test_store_local_creates_mirror_copy(self, temp_transport_paths: dict) -> None:
        """Creates mirror copy when local_path differs from backup_path."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            enable_s3=False,
        )

        backup_id = "TEST_BACKUP_008"
        data = b"mirror test data"

        await service.store(backup_id, data, StorageTier.LOCAL)

        # Verify both locations have the file
        primary_path = temp_transport_paths["backup"] / f"{backup_id}.backup"
        mirror_path = temp_transport_paths["local"] / f"{backup_id}.backup"

        assert primary_path.exists()
        assert mirror_path.exists()
        assert primary_path.read_bytes() == data
        assert mirror_path.read_bytes() == data

    @pytest.mark.asyncio
    async def test_store_local_mirror_copy_error_handled(self, temp_transport_paths: dict) -> None:
        """Handles mirror copy failure gracefully."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            enable_s3=False,
        )

        backup_id = "TEST_BACKUP_009"
        data = b"mirror error test"

        # Make local path read-only to cause mirror failure
        temp_transport_paths["local"].chmod(0o000)

        try:
            # Should still succeed even if mirror fails
            path = await service.store(backup_id, data, StorageTier.LOCAL)

            # Primary should exist
            assert Path(path).exists()
            assert Path(path).read_bytes() == data
        finally:
            # Restore permissions
            temp_transport_paths["local"].chmod(0o755)


class TestRetrieveOperations:
    """Tests for backup retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_local_tier(self, transport_service: TransportService) -> None:
        """Retrieves backup from LOCAL tier."""
        backup_id = "TEST_RETRIEVE_001"
        data = b"retrieve test data"

        # Store first
        await transport_service.store(backup_id, data, StorageTier.LOCAL)

        # Retrieve
        retrieved = await transport_service.retrieve(backup_id, StorageTier.LOCAL)

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_retrieve_local_returns_none_when_not_found(
        self, transport_service: TransportService
    ) -> None:
        """Returns None when backup not found in LOCAL tier."""
        result = await transport_service.retrieve("NONEXISTENT", StorageTier.LOCAL)

        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_network_tier(self, transport_service: TransportService) -> None:
        """Retrieves backup from NETWORK tier."""
        backup_id = "TEST_RETRIEVE_002"
        data = b"network retrieve data"

        await transport_service.store(backup_id, data, StorageTier.NETWORK)
        retrieved = await transport_service.retrieve(backup_id, StorageTier.NETWORK)

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_retrieve_network_returns_none_when_not_found(
        self, transport_service: TransportService
    ) -> None:
        """Returns None when backup not found in NETWORK tier."""
        result = await transport_service.retrieve("NONEXISTENT", StorageTier.NETWORK)

        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_network_returns_none_when_unavailable(
        self, temp_transport_paths: dict
    ) -> None:
        """Returns None when network storage unavailable."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            network_path=None,
            enable_s3=False,
        )

        result = await service.retrieve("TEST", StorageTier.NETWORK)

        assert result is None

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_retrieve_cloud_tier_from_s3(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Retrieves backup from CLOUD tier (S3)."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})

        # Mock S3 get_object response
        mock_body = Mock()
        mock_body.read = Mock(return_value=b"s3 backup data")
        mock_s3.get_object = Mock(return_value={"Body": mock_body})

        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        result = await service.retrieve("TEST_S3", StorageTier.CLOUD)

        assert result == b"s3 backup data"
        mock_s3.get_object.assert_called_once()

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_retrieve_archive_tier_from_glacier(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Retrieves backup from ARCHIVE tier (Glacier)."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})

        mock_body = Mock()
        mock_body.read = Mock(return_value=b"glacier data")
        mock_s3.get_object = Mock(return_value={"Body": mock_body})

        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        result = await service.retrieve("TEST_GLACIER", StorageTier.ARCHIVE)

        assert result == b"glacier data"
        # Verify called with archive prefix
        call_args = mock_s3.get_object.call_args
        assert "archive/" in call_args.kwargs["Key"]

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_retrieve_s3_returns_none_on_error(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Returns None when S3 retrieval fails."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.get_object.side_effect = Exception("S3 error")
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        result = await service.retrieve("TEST_ERROR", StorageTier.CLOUD)

        assert result is None


class TestDeleteOperations:
    """Tests for backup deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_local_tier(self, transport_service: TransportService) -> None:
        """Deletes backup from LOCAL tier."""
        backup_id = "TEST_DELETE_001"
        data = b"delete test data"

        # Store first
        path = await transport_service.store(backup_id, data, StorageTier.LOCAL)
        assert Path(path).exists()

        # Delete
        success = await transport_service.delete(backup_id, StorageTier.LOCAL)

        assert success is True
        assert not Path(path).exists()

    @pytest.mark.asyncio
    async def test_delete_network_tier(self, transport_service: TransportService) -> None:
        """Deletes backup from NETWORK tier."""
        backup_id = "TEST_DELETE_002"
        data = b"network delete data"

        path = await transport_service.store(backup_id, data, StorageTier.NETWORK)
        assert Path(path).exists()

        success = await transport_service.delete(backup_id, StorageTier.NETWORK)

        assert success is True
        assert not Path(path).exists()

    @pytest.mark.asyncio
    @skip_if_no_boto3
    @patch("boto3.client")
    async def test_delete_cloud_tier_from_s3(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Deletes backup from CLOUD tier (S3)."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.delete_object = Mock(return_value={})
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        success = await service.delete("TEST_DELETE_S3", StorageTier.CLOUD)

        assert success is True
        mock_s3.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, temp_transport_paths: dict) -> None:
        """Returns False when deletion fails."""
        # Create service with invalid permissions
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            enable_s3=False,
        )

        # Create a backup file
        backup_id = "TEST_DELETE_ERROR"
        backup_file = temp_transport_paths["backup"] / f"{backup_id}.backup"
        backup_file.write_bytes(b"test data")

        # Make file read-only and directory read-only to cause deletion failure
        backup_file.chmod(0o000)
        temp_transport_paths["backup"].chmod(0o000)

        try:
            success = await service.delete(backup_id, StorageTier.LOCAL)
            assert success is False
        finally:
            # Restore permissions for cleanup
            temp_transport_paths["backup"].chmod(0o755)
            backup_file.chmod(0o644)


class TestUploadToS3:
    """Tests for S3 upload operations."""

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_upload_to_s3_success(self, mock_boto3_client, temp_transport_paths: dict) -> None:
        """Uploads local backup to S3 successfully."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.upload_file = Mock(return_value=None)
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        # Create local backup file
        backup_id = "TEST_UPLOAD"
        backup_file = temp_transport_paths["backup"] / f"{backup_id}.backup"
        backup_file.write_bytes(b"upload test data")

        service.upload_to_s3(backup_id)

        mock_s3.upload_file.assert_called_once()

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_upload_to_s3_not_found(self, mock_boto3_client, temp_transport_paths: dict) -> None:
        """Handles missing local backup gracefully."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.upload_file = Mock()
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        # Don't create local file
        service.upload_to_s3("NONEXISTENT")

        # Should not call upload_file
        mock_s3.upload_file.assert_not_called()

    def test_upload_to_s3_when_s3_unavailable(self, temp_transport_paths: dict) -> None:
        """Does nothing when S3 client unavailable."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket=None,
            enable_s3=False,
        )

        # Should not raise
        service.upload_to_s3("TEST")

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_upload_to_s3_error_handling(
        self, mock_boto3_client, temp_transport_paths: dict
    ) -> None:
        """Handles S3 upload errors gracefully."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_s3.upload_file.side_effect = Exception("Upload error")
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        # Create local backup file
        backup_id = "TEST_ERROR"
        backup_file = temp_transport_paths["backup"] / f"{backup_id}.backup"
        backup_file.write_bytes(b"error test data")

        # Should not raise
        service.upload_to_s3(backup_id)


class TestStorageProperties:
    """Tests for storage availability properties."""

    def test_has_network_storage_true(self, transport_service: TransportService) -> None:
        """has_network_storage is True when network path configured."""
        assert transport_service.has_network_storage is True

    def test_has_network_storage_false(self, temp_transport_paths: dict) -> None:
        """has_network_storage is False when network path is None."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            network_path=None,
            enable_s3=False,
        )

        assert service.has_network_storage is False

    @skip_if_no_boto3
    @patch("boto3.client")
    def test_has_cloud_storage_true(self, mock_boto3_client, temp_transport_paths: dict) -> None:
        """has_cloud_storage is True when S3 client initialized."""
        mock_s3 = Mock()
        mock_s3.head_bucket = Mock(return_value={})
        mock_boto3_client.return_value = mock_s3

        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket="test-bucket",
            enable_s3=True,
        )

        assert service.has_cloud_storage is True

    def test_has_cloud_storage_false(self, temp_transport_paths: dict) -> None:
        """has_cloud_storage is False when S3 client is None."""
        service = TransportService(
            local_path=temp_transport_paths["local"],
            backup_path=temp_transport_paths["backup"],
            s3_bucket=None,
            enable_s3=False,
        )

        assert service.has_cloud_storage is False
