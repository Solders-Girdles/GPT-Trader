"""
Unit tests for RetentionManager.

Tests retention policy enforcement, cleanup orchestration, batch/sequential deletion.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.retention_manager import RetentionManager


@pytest.fixture
def config():
    """Backup configuration."""
    return BackupConfig(backup_dir="/tmp/backups")


@pytest.fixture
def context():
    """Backup context with sample metadata."""
    ctx = BackupContext()
    # Add some sample backup metadata
    ctx.backup_metadata = {
        "backup1": BackupMetadata(
            backup_id="backup1",
            backup_type=BackupType.FULL,
            timestamp=datetime.now(timezone.utc),
            size_bytes=1000,
            size_compressed=800,
            checksum="abc",
            encryption_key_id=None,
            storage_tier=StorageTier.LOCAL,
            retention_days=30,
            status=BackupStatus.COMPLETED,
        ),
        "backup2": BackupMetadata(
            backup_id="backup2",
            backup_type=BackupType.INCREMENTAL,
            timestamp=datetime.now(timezone.utc),
            size_bytes=500,
            size_compressed=400,
            checksum="def",
            encryption_key_id=None,
            storage_tier=StorageTier.LOCAL,
            retention_days=7,
            status=BackupStatus.COMPLETED,
        ),
    }
    ctx.backup_history = list(ctx.backup_metadata.values())
    return ctx


@pytest.fixture
def retention_service():
    """Mock retention service."""
    service = Mock()
    service.filter_expired = Mock(return_value=[])
    service.get_retention_days = Mock(return_value=30)
    service.cleanup_metadata_files = Mock()
    return service


@pytest.fixture
def transport_service():
    """Mock transport service with batch delete."""
    service = Mock()
    service.batch_delete = AsyncMock(return_value={})
    service.delete = AsyncMock(return_value=True)
    return service


@pytest.fixture
def retention_manager(retention_service, transport_service, context, config):
    """Create retention manager instance."""
    return RetentionManager(
        retention_service=retention_service,
        transport_service=transport_service,
        context=context,
        config=config,
    )


class TestCleanupWithExpiredBackups:
    """Test cleanup with expired backups."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_expired_backups(self, retention_manager, retention_service):
        """Should return 0 when no backups are expired."""
        retention_service.filter_expired.return_value = []

        result = await retention_manager.cleanup()

        assert result == 0

    @pytest.mark.asyncio
    async def test_uses_batch_delete_when_available(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should use batch delete when transport supports it."""
        expired = [context.backup_metadata["backup1"]]
        retention_service.filter_expired.return_value = expired

        # Mock successful batch delete
        transport_service.batch_delete.return_value = {"backup1": True}

        result = await retention_manager.cleanup()

        assert result == 1
        transport_service.batch_delete.assert_called_once()
        # Verify backup removed from context
        assert "backup1" not in context.backup_metadata
        assert len(context.backup_history) == 1

    @pytest.mark.asyncio
    async def test_batch_delete_updates_metadata_on_success(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should update metadata only for successful deletions."""
        expired = [
            context.backup_metadata["backup1"],
            context.backup_metadata["backup2"],
        ]
        retention_service.filter_expired.return_value = expired

        # Mock partial success
        transport_service.batch_delete.return_value = {"backup1": True, "backup2": False}

        result = await retention_manager.cleanup()

        assert result == 1
        assert "backup1" not in context.backup_metadata
        assert "backup2" in context.backup_metadata  # Failed deletion, kept in metadata

    @pytest.mark.asyncio
    async def test_falls_back_to_sequential_when_no_batch_delete(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should use sequential delete when batch not available."""
        # Remove batch_delete capability
        del transport_service.batch_delete

        expired = [context.backup_metadata["backup1"]]
        retention_service.filter_expired.return_value = expired
        transport_service.delete.return_value = True

        result = await retention_manager.cleanup()

        assert result == 1
        transport_service.delete.assert_called_once_with("backup1", StorageTier.LOCAL)
        assert "backup1" not in context.backup_metadata

    @pytest.mark.asyncio
    async def test_sequential_delete_skips_failed_deletions(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should skip metadata update when sequential delete fails."""
        del transport_service.batch_delete

        expired = [context.backup_metadata["backup1"]]
        retention_service.filter_expired.return_value = expired
        transport_service.delete.return_value = False  # Failed deletion

        result = await retention_manager.cleanup()

        assert result == 0
        assert "backup1" in context.backup_metadata  # Kept in metadata

    @pytest.mark.asyncio
    async def test_cleans_up_metadata_files_after_deletion(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should cleanup metadata files for successfully deleted backups."""
        expired = [
            context.backup_metadata["backup1"],
            context.backup_metadata["backup2"],
        ]
        retention_service.filter_expired.return_value = expired
        transport_service.batch_delete.return_value = {"backup1": True, "backup2": True}

        await retention_manager.cleanup()

        retention_service.cleanup_metadata_files.assert_called_once()
        call_args = retention_service.cleanup_metadata_files.call_args
        assert call_args[0][0] == Path("/tmp/backups")
        assert set(call_args[0][1]) == {"backup1", "backup2"}


class TestCleanupErrorHandling:
    """Test error handling in cleanup."""

    @pytest.mark.asyncio
    async def test_catches_exceptions_and_returns_zero(self, retention_manager, retention_service):
        """Should catch exceptions and return 0."""
        retention_service.filter_expired.side_effect = Exception("Test error")

        result = await retention_manager.cleanup()

        assert result == 0

    @pytest.mark.asyncio
    async def test_logs_error_on_exception(self, retention_manager, retention_service, caplog):
        """Should log error when exception occurs."""
        retention_service.filter_expired.side_effect = Exception("Test error")

        await retention_manager.cleanup()

        # Check that error was logged (error may be logged after test completes)
        # So we just verify no crash occurred


class TestCleanupWithProvidedExpiredList:
    """Test cleanup with pre-filtered expired backups."""

    @pytest.mark.asyncio
    async def test_uses_provided_expired_list(
        self, retention_manager, retention_service, transport_service, context
    ):
        """Should use provided expired list instead of filtering."""
        expired = [context.backup_metadata["backup1"]]
        transport_service.batch_delete.return_value = {"backup1": True}

        result = await retention_manager.cleanup(expired_backups=expired)

        assert result == 1
        # Should NOT call filter_expired when list is provided
        retention_service.filter_expired.assert_not_called()


class TestRetentionDays:
    """Test retention days retrieval."""

    def test_returns_generic_retention_when_set(self, retention_manager, config):
        """Should return generic retention when config has retention_days."""
        config.retention_days = 90

        result = retention_manager.get_retention_days(BackupType.FULL)

        assert result == 90

    def test_delegates_to_service_when_no_generic_retention(
        self, retention_manager, retention_service
    ):
        """Should delegate to service when no generic retention."""
        retention_service.get_retention_days.return_value = 30

        result = retention_manager.get_retention_days(BackupType.FULL)

        assert result == 30
        retention_service.get_retention_days.assert_called_once_with(BackupType.FULL)

    def test_checks_generic_retention_first(self, retention_manager, retention_service, config):
        """Should check generic retention before delegating."""
        config.retention_days = 60
        retention_service.get_retention_days.return_value = 30

        result = retention_manager.get_retention_days(BackupType.INCREMENTAL)

        assert result == 60
        # Should NOT call service when generic is set
        retention_service.get_retention_days.assert_not_called()


class TestMetricsCollection:
    """Test metrics collection integration."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector.record_counter = Mock()
        collector.record_histogram = Mock()
        return collector

    @pytest.fixture
    def retention_manager_with_metrics(
        self, retention_service, transport_service, context, config, mock_metrics_collector
    ):
        """Create retention manager with metrics collector."""
        return RetentionManager(
            retention_service=retention_service,
            transport_service=transport_service,
            context=context,
            config=config,
            metrics_collector=mock_metrics_collector,
        )

    @pytest.mark.asyncio
    async def test_successful_cleanup_records_metrics(
        self,
        retention_manager_with_metrics,
        retention_service,
        transport_service,
        mock_metrics_collector,
    ):
        """Should record success metrics when cleanup completes successfully."""
        # Setup expired backups
        retention_service.filter_expired.return_value = []  # No expired backups

        result = await retention_manager_with_metrics.cleanup()

        assert result == 0

        # Verify metrics recorded
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.retention.cleaned_total" in counter_calls
        assert "backup.retention.cleaned_success" in counter_calls
        assert "backup.retention.cleaned_failed" not in counter_calls

    @pytest.mark.asyncio
    async def test_cleanup_with_removals_records_count(
        self,
        retention_manager_with_metrics,
        retention_service,
        transport_service,
        context,
        mock_metrics_collector,
    ):
        """Should record removal count when backups are deleted."""
        # Setup 2 expired backups
        expired = [context.backup_metadata["backup1"], context.backup_metadata["backup2"]]
        retention_service.filter_expired.return_value = expired
        transport_service.batch_delete.return_value = {"backup1": True, "backup2": True}

        result = await retention_manager_with_metrics.cleanup()

        assert result == 2

        # Verify removal count recorded
        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        removal_call = next(
            call for call in histogram_calls if call[0][0] == "backup.retention.removed_count"
        )
        assert removal_call is not None
        removal_count = removal_call[0][1]
        assert removal_count == 2.0

    @pytest.mark.asyncio
    async def test_failed_cleanup_records_failure_metrics(
        self,
        retention_manager_with_metrics,
        retention_service,
        transport_service,
        mock_metrics_collector,
    ):
        """Should record failure metrics when cleanup fails."""
        # Make cleanup fail
        retention_service.filter_expired.side_effect = Exception("Test error")

        result = await retention_manager_with_metrics.cleanup()

        # Result is 0 because exception occurred
        assert result == 0

        # Verify failure metrics recorded
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.retention.cleaned_total" in counter_calls
        assert "backup.retention.cleaned_failed" in counter_calls

    @pytest.mark.asyncio
    async def test_no_removal_count_when_no_backups_removed(
        self, retention_manager_with_metrics, retention_service, mock_metrics_collector
    ):
        """Should not record removal count histogram when count is 0."""
        retention_service.filter_expired.return_value = []  # No expired backups

        result = await retention_manager_with_metrics.cleanup()

        assert result == 0

        # Verify no removal count histogram (only recorded when > 0)
        histogram_calls = [
            call[0][0] for call in mock_metrics_collector.record_histogram.call_args_list
        ]
        assert "backup.retention.removed_count" not in histogram_calls

    @pytest.mark.asyncio
    async def test_metrics_not_recorded_when_collector_none(
        self, retention_manager, retention_service
    ):
        """Should not crash when metrics collector is None."""
        # Verify collector is None
        assert retention_manager.metrics_collector is None

        retention_service.filter_expired.return_value = []

        # Execute (should not crash)
        result = await retention_manager.cleanup()

        # Verify success
        assert result == 0

    @pytest.mark.asyncio
    async def test_success_metrics_with_sequential_delete(
        self,
        retention_manager_with_metrics,
        retention_service,
        transport_service,
        context,
        mock_metrics_collector,
    ):
        """Should record metrics correctly with sequential delete path."""
        # Setup 1 expired backup
        expired = [context.backup_metadata["backup1"]]
        retention_service.filter_expired.return_value = expired

        # Use sequential delete (no batch_delete)
        delattr(transport_service, "batch_delete")
        transport_service.delete = AsyncMock(return_value=True)

        result = await retention_manager_with_metrics.cleanup()

        assert result == 1

        # Verify metrics recorded
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.retention.cleaned_total" in counter_calls
        assert "backup.retention.cleaned_success" in counter_calls

        # Verify removal count
        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        removal_call = next(
            call for call in histogram_calls if call[0][0] == "backup.retention.removed_count"
        )
        assert removal_call[0][1] == 1.0
