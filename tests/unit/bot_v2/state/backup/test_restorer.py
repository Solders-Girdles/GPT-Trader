"""Unit tests for BackupRestorer.

Tests restoration logic in isolation:
- Retrieval and verification
- Decryption and decompression
- State application with batch operations
- Latest backup selection
"""

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.restorer import BackupRestorer


@pytest.fixture
def backup_config():
    """Minimal backup configuration."""
    return BackupConfig(
        enable_encryption=True,
        enable_compression=True,
    )


@pytest.fixture
def backup_context():
    """Shared backup context."""
    context = BackupContext()
    # Add sample backup history
    context.backup_history = [
        BackupMetadata(
            backup_id="FULL_20250101_120000",
            backup_type=BackupType.FULL,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size_bytes=1000,
            size_compressed=500,
            checksum="abc123",
            encryption_key_id="key_1",
            storage_tier=StorageTier.LOCAL,
            retention_days=90,
            status=BackupStatus.COMPLETED,
        ),
        BackupMetadata(
            backup_id="INC_20250101_130000",
            backup_type=BackupType.INCREMENTAL,
            timestamp=datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            size_bytes=500,
            size_compressed=250,
            checksum="def456",
            encryption_key_id="key_2",
            storage_tier=StorageTier.LOCAL,
            retention_days=7,
            status=BackupStatus.VERIFIED,
        ),
    ]
    return context


@pytest.fixture
def mock_state_manager():
    """Mock state manager with batch operations."""
    manager = Mock()
    manager.set_state = AsyncMock(return_value=True)
    manager.batch_set_state = AsyncMock(return_value=5)  # Returns count
    return manager


@pytest.fixture
def mock_metadata_manager(backup_context):
    """Mock metadata manager."""
    manager = Mock()
    manager.find_metadata = Mock(
        side_effect=lambda backup_id: next(
            (b for b in backup_context.backup_history if b.backup_id == backup_id), None
        )
    )
    return manager


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service."""
    service = Mock()
    service.decrypt = Mock(return_value=b"decrypted_data")
    return service


@pytest.fixture
def mock_compression_service():
    """Mock compression service."""
    service = Mock()
    service.decompress = Mock(
        return_value=json.dumps(
            {"state": {"position:BTC": {"qty": 1.5}, "portfolio_current": {"cash": 10000}}}
        ).encode()
    )
    return service


@pytest.fixture
def mock_transport_service():
    """Mock transport service."""
    service = Mock()

    # Create valid backup payload
    payload = {"state": {"position:BTC": {"qty": 1.5}}}
    payload_bytes = json.dumps(payload).encode()
    checksum = hashlib.sha256(payload_bytes).hexdigest()

    service.retrieve = AsyncMock(return_value=payload_bytes)
    service.checksum = checksum
    return service


@pytest.fixture
def backup_restorer(
    mock_state_manager,
    backup_config,
    backup_context,
    mock_metadata_manager,
    mock_encryption_service,
    mock_compression_service,
    mock_transport_service,
):
    """BackupRestorer instance with mocked dependencies."""
    return BackupRestorer(
        state_manager=mock_state_manager,
        config=backup_config,
        context=backup_context,
        metadata_manager=mock_metadata_manager,
        encryption_service=mock_encryption_service,
        compression_service=mock_compression_service,
        transport_service=mock_transport_service,
    )


class TestBackupRestoration:
    """Tests for backup restoration workflow."""

    @pytest.mark.asyncio
    async def test_restores_backup_successfully(self, backup_restorer, backup_context):
        """Restores backup with complete pipeline."""
        backup_id = "FULL_20250101_120000"

        # Update transport to return valid encrypted/compressed data
        payload = {"state": {"position:BTC": {"qty": 1.5}}}
        compressed = json.dumps(payload).encode()
        encrypted = b"encrypted:" + compressed

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=encrypted)
        backup_restorer.encryption_service.decrypt = Mock(return_value=compressed)
        backup_restorer.compression_service.decompress = Mock(return_value=compressed)

        # Update metadata with correct checksum
        metadata = backup_context.backup_history[0]
        metadata.checksum = hashlib.sha256(encrypted).hexdigest()

        result = await backup_restorer.restore_from_backup_internal(backup_id)

        assert result == {"position:BTC": {"qty": 1.5}}
        assert backup_context.last_restored_payload == {"position:BTC": {"qty": 1.5}}

    @pytest.mark.asyncio
    async def test_raises_error_for_missing_backup(self, backup_restorer):
        """Raises FileNotFoundError for non-existent backup."""
        with pytest.raises(FileNotFoundError, match="Backup .* not found"):
            await backup_restorer.restore_from_backup_internal("MISSING_BACKUP")

    @pytest.mark.asyncio
    async def test_verifies_checksum(self, backup_restorer, backup_context):
        """Verifies backup checksum before restoration."""
        backup_id = "FULL_20250101_120000"

        # Return data with wrong checksum
        wrong_data = b"corrupted_data"
        backup_restorer.transport_service.retrieve = AsyncMock(return_value=wrong_data)

        with pytest.raises(ValueError, match="checksum mismatch"):
            await backup_restorer.restore_from_backup_internal(backup_id)

    @pytest.mark.asyncio
    async def test_decrypts_encrypted_backup(
        self, backup_restorer, backup_context, mock_encryption_service
    ):
        """Decrypts backup when encryption key is present."""
        backup_id = "FULL_20250101_120000"
        metadata = backup_context.backup_history[0]

        # Setup valid encrypted payload
        payload = {"state": {"key": "value"}}
        decrypted = json.dumps(payload).encode()
        encrypted = b"encrypted_version"

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=encrypted)
        backup_restorer.encryption_service.decrypt = Mock(return_value=decrypted)
        backup_restorer.compression_service.decompress = Mock(return_value=decrypted)

        # Update checksum to match encrypted data
        metadata.checksum = hashlib.sha256(encrypted).hexdigest()

        await backup_restorer.restore_from_backup_internal(backup_id)

        # Verify decryption was called
        mock_encryption_service.decrypt.assert_called_once_with(encrypted)

    @pytest.mark.asyncio
    async def test_decompresses_compressed_backup(
        self, backup_restorer, backup_context, mock_compression_service
    ):
        """Decompresses backup when compression is enabled."""
        backup_id = "FULL_20250101_120000"
        metadata = backup_context.backup_history[0]
        metadata.size_compressed = 500  # Indicates compression

        # Setup valid compressed payload
        payload = {"state": {"key": "value"}}
        decompressed = json.dumps(payload).encode()
        compressed = b"compressed_version"

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=compressed)
        backup_restorer.encryption_service.decrypt = Mock(return_value=compressed)
        backup_restorer.compression_service.decompress = Mock(return_value=decompressed)

        # Update checksum
        metadata.checksum = hashlib.sha256(compressed).hexdigest()

        await backup_restorer.restore_from_backup_internal(backup_id)

        # Verify decompression was called
        mock_compression_service.decompress.assert_called_once()

    @pytest.mark.asyncio
    async def test_applies_state_to_state_manager(
        self, backup_restorer, backup_context, mock_state_manager
    ):
        """Applies restored state to state manager."""
        backup_id = "FULL_20250101_120000"

        # Setup valid payload
        payload = {"state": {"position:BTC": {"qty": 1.5}}}
        payload_bytes = json.dumps(payload).encode()

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=payload_bytes)
        backup_restorer.encryption_service.decrypt = Mock(return_value=payload_bytes)
        backup_restorer.compression_service.decompress = Mock(return_value=payload_bytes)

        # Update checksum
        metadata = backup_context.backup_history[0]
        metadata.checksum = hashlib.sha256(payload_bytes).hexdigest()

        await backup_restorer.restore_from_backup_internal(backup_id, apply_state=True)

        # Verify state was applied
        mock_state_manager.batch_set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_state_application_when_disabled(
        self, backup_restorer, backup_context, mock_state_manager
    ):
        """Does not apply state when apply_state=False."""
        backup_id = "FULL_20250101_120000"

        # Setup valid payload
        payload = {"state": {"position:BTC": {"qty": 1.5}}}
        payload_bytes = json.dumps(payload).encode()

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=payload_bytes)
        backup_restorer.encryption_service.decrypt = Mock(return_value=payload_bytes)
        backup_restorer.compression_service.decompress = Mock(return_value=payload_bytes)

        # Update checksum
        metadata = backup_context.backup_history[0]
        metadata.checksum = hashlib.sha256(payload_bytes).hexdigest()

        result = await backup_restorer.restore_from_backup_internal(backup_id, apply_state=False)

        # Verify state was NOT applied
        mock_state_manager.batch_set_state.assert_not_called()
        mock_state_manager.set_state.assert_not_called()

        # But result was returned
        assert result == {"position:BTC": {"qty": 1.5}}


class TestLatestBackupRestoration:
    """Tests for restore_latest_backup."""

    @pytest.mark.asyncio
    async def test_restores_latest_backup(self, backup_restorer, backup_context):
        """Restores most recent backup."""
        # Setup valid payload for latest backup
        payload = {"state": {"key": "value"}}
        payload_bytes = json.dumps(payload).encode()

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=payload_bytes)
        backup_restorer.encryption_service.decrypt = Mock(return_value=payload_bytes)
        backup_restorer.compression_service.decompress = Mock(return_value=payload_bytes)

        # Update checksum for latest backup
        latest = backup_context.backup_history[-1]
        latest.checksum = hashlib.sha256(payload_bytes).hexdigest()

        result = await backup_restorer.restore_latest_backup()

        assert result is True

    @pytest.mark.asyncio
    async def test_filters_by_backup_type(self, backup_restorer, backup_context):
        """Filters backups by type when specified."""
        # Setup valid payload
        payload = {"state": {"key": "value"}}
        payload_bytes = json.dumps(payload).encode()

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=payload_bytes)
        backup_restorer.encryption_service.decrypt = Mock(return_value=payload_bytes)
        backup_restorer.compression_service.decompress = Mock(return_value=payload_bytes)

        # Update checksum for FULL backup
        full_backup = backup_context.backup_history[0]
        full_backup.checksum = hashlib.sha256(payload_bytes).hexdigest()

        result = await backup_restorer.restore_latest_backup(backup_type=BackupType.FULL)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_backups(self, backup_restorer, backup_context):
        """Returns False when no valid backups exist."""
        backup_context.backup_history = []

        result = await backup_restorer.restore_latest_backup()

        assert result is False

    @pytest.mark.asyncio
    async def test_skips_failed_backups(self, backup_restorer, backup_context):
        """Only considers COMPLETED or VERIFIED backups."""
        # Mark all backups as FAILED
        for backup in backup_context.backup_history:
            backup.status = BackupStatus.FAILED

        result = await backup_restorer.restore_latest_backup()

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_restoration_errors(self, backup_restorer, backup_context):
        """Returns False when restoration fails."""
        # Make transport service fail
        backup_restorer.transport_service.retrieve = AsyncMock(
            side_effect=Exception("Transport error")
        )

        result = await backup_restorer.restore_latest_backup()

        assert result is False


class TestBatchStateApplication:
    """Tests for batch state restoration."""

    @pytest.mark.asyncio
    async def test_uses_batch_operations_when_available(self, backup_restorer, mock_state_manager):
        """Uses batch_set_state for better performance."""
        data = {"position:BTC": {"qty": 1.5}, "order:123": {"status": "filled"}}

        result = await backup_restorer._restore_data_to_state(data)

        # Verify batch operation was used
        mock_state_manager.batch_set_state.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_falls_back_to_sequential_when_batch_unavailable(self):
        """Falls back to sequential set_state when batch not available."""
        from bot_v2.state.state_manager import StateCategory

        # Create state manager without batch_set_state
        state_manager = Mock()
        state_manager.set_state = AsyncMock(return_value=True)
        # Explicitly set batch_set_state to False to trigger hasattr check
        state_manager.batch_set_state = None

        restorer = BackupRestorer(
            state_manager=state_manager,
            config=BackupConfig(),
            context=BackupContext(),
            metadata_manager=Mock(),
            encryption_service=Mock(),
            compression_service=Mock(),
            transport_service=Mock(),
        )

        # Remove batch_set_state attribute entirely to trigger fallback
        delattr(state_manager, "batch_set_state")

        data = {"position:BTC": {"qty": 1.5}, "order:123": {"status": "filled"}}

        result = await restorer._restore_data_to_state(data)

        # Verify sequential operations were used
        assert state_manager.set_state.call_count == 2
        assert result is True

    @pytest.mark.asyncio
    async def test_categorizes_state_correctly(self, backup_restorer, mock_state_manager):
        """Assigns correct StateCategory based on key patterns."""
        from bot_v2.state.state_manager import StateCategory

        data = {
            "position:BTC": {"qty": 1.5},  # HOT
            "order:123": {"status": "filled"},  # HOT
            "ml_model:predictor": {"weights": []},  # WARM
            "config:trading": {"max_size": 100},  # WARM
        }

        await backup_restorer._restore_data_to_state(data)

        # Verify batch_set_state was called with categorized items
        call_args = mock_state_manager.batch_set_state.call_args[0][0]

        # Check categories
        assert call_args["position:BTC"][1] == StateCategory.HOT
        assert call_args["order:123"][1] == StateCategory.HOT
        assert call_args["ml_model:predictor"][1] == StateCategory.WARM
        assert call_args["config:trading"][1] == StateCategory.WARM

    @pytest.mark.asyncio
    async def test_handles_non_numeric_batch_return(self, backup_restorer):
        """Handles batch_set_state returning non-numeric value (e.g., Mock)."""
        # Mock batch_set_state to return a Mock object
        backup_restorer.state_manager.batch_set_state = AsyncMock(return_value=Mock())

        data = {"position:BTC": {"qty": 1.5}}

        result = await backup_restorer._restore_data_to_state(data)

        # Should assume success if items were present
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, backup_restorer):
        """Returns False when state application fails."""
        # Make batch_set_state raise exception
        backup_restorer.state_manager.batch_set_state = AsyncMock(
            side_effect=Exception("State error")
        )

        data = {"position:BTC": {"qty": 1.5}}

        result = await backup_restorer._restore_data_to_state(data)

        assert result is False


class TestPayloadValidation:
    """Tests for backup payload validation."""

    @pytest.mark.asyncio
    async def test_raises_error_for_invalid_payload(self, backup_restorer, backup_context):
        """Raises ValueError for non-dict payload."""
        backup_id = "FULL_20250101_120000"

        # Return invalid payload (list wrapped in "state" key will fail)
        invalid_payload = json.dumps({"state": ["not", "a", "dict"]}).encode()

        backup_restorer.transport_service.retrieve = AsyncMock(return_value=invalid_payload)
        backup_restorer.encryption_service.decrypt = Mock(return_value=invalid_payload)
        backup_restorer.compression_service.decompress = Mock(return_value=invalid_payload)

        # Update checksum
        metadata = backup_context.backup_history[0]
        metadata.checksum = hashlib.sha256(invalid_payload).hexdigest()

        with pytest.raises(ValueError, match="not a mapping"):
            await backup_restorer.restore_from_backup_internal(backup_id)

    @pytest.mark.asyncio
    async def test_handles_missing_backup_payload(self, backup_restorer):
        """Raises FileNotFoundError when backup payload is missing."""
        backup_id = "FULL_20250101_120000"

        # Transport returns None
        backup_restorer.transport_service.retrieve = AsyncMock(return_value=None)

        with pytest.raises(FileNotFoundError, match="payload missing"):
            await backup_restorer.restore_from_backup_internal(backup_id)
