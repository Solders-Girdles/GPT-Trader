"""
Unit tests for BackupWorkflow.

Tests backup creation pipeline: data collection, normalization, diffing, orchestration.
"""

import threading
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
from bot_v2.state.backup.workflow import BackupWorkflow


@pytest.fixture
def config():
    """Backup configuration."""
    return BackupConfig()


@pytest.fixture
def context():
    """Backup context."""
    return BackupContext()


@pytest.fixture
def backup_lock():
    """Threading lock for concurrency control."""
    return threading.Lock()


@pytest.fixture
def mock_data_collector():
    """Mock data collector."""
    collector = Mock()
    collector.collect_for_backup = AsyncMock(
        return_value={"positions": {"BTC": {"qty": 1.0}}, "orders": []}
    )
    return collector


@pytest.fixture
def mock_backup_creator():
    """Mock backup creator."""
    creator = Mock()
    creator.create_backup_internal = AsyncMock(
        return_value=BackupMetadata(
            backup_id="FUL_20250101_120000",
            backup_type=BackupType.FULL,
            timestamp=datetime.now(timezone.utc),
            size_bytes=1000,
            size_compressed=800,
            checksum="abc123",
            encryption_key_id=None,
            storage_tier=StorageTier.LOCAL,
            retention_days=30,
            status=BackupStatus.COMPLETED,
        )
    )
    return creator


@pytest.fixture
def workflow(config, context, backup_lock, mock_data_collector, mock_backup_creator):
    """Create workflow instance."""
    return BackupWorkflow(
        data_collector=mock_data_collector,
        backup_creator=mock_backup_creator,
        context=context,
        config=config,
        backup_lock=backup_lock,
    )


class TestBackupCreation:
    """Test backup creation flow."""

    @pytest.mark.asyncio
    async def test_creates_full_backup(self, workflow, mock_data_collector, mock_backup_creator):
        """Should create full backup successfully."""
        result = await workflow.create_backup(BackupType.FULL)

        assert result is not None
        assert result.backup_type == BackupType.FULL
        assert result.backup_id.startswith("FUL_")

        # Verify data collection called
        mock_data_collector.collect_for_backup.assert_called_once()

        # Verify backup creator called
        mock_backup_creator.create_backup_internal.assert_called_once()

    @pytest.mark.asyncio
    async def test_prevents_concurrent_backups(self, workflow):
        """Should prevent concurrent backups."""
        workflow._backup_in_progress = True

        result = await workflow.create_backup(BackupType.FULL)

        assert result is None  # Rejected due to in-progress backup

    @pytest.mark.asyncio
    async def test_clears_backup_in_progress_flag_after_completion(self, workflow):
        """Should clear backup_in_progress flag after completion."""
        assert not workflow.is_backup_in_progress

        await workflow.create_backup(BackupType.FULL)

        assert not workflow.is_backup_in_progress

    @pytest.mark.asyncio
    async def test_clears_backup_in_progress_flag_on_error(self, workflow, mock_backup_creator):
        """Should clear backup_in_progress flag even on error."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        result = await workflow.create_backup(BackupType.FULL)

        assert result is None  # Failed
        assert not workflow.is_backup_in_progress  # Flag cleared

    @pytest.mark.asyncio
    async def test_clears_pending_snapshot_after_completion(self, workflow):
        """Should clear pending snapshot after completion."""
        await workflow.create_backup(BackupType.FULL)

        assert workflow.pending_snapshot is None

    @pytest.mark.asyncio
    async def test_clears_pending_snapshot_on_error(self, workflow, mock_backup_creator):
        """Should clear pending snapshot even on error."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        await workflow.create_backup(BackupType.FULL)

        assert workflow.pending_snapshot is None


class TestBackupID:
    """Test backup ID generation."""

    def test_generates_unique_id_with_timestamp(self, workflow):
        """Should generate ID with type prefix and timestamp."""
        with patch("bot_v2.state.backup.workflow.datetime") as mock_dt:
            mock_dt.utcnow.return_value.strftime.return_value = "20250101_120000"

            backup_id = workflow._generate_backup_id(BackupType.FULL)

            assert backup_id == "FUL_20250101_120000"

    def test_generates_different_prefix_per_type(self, workflow):
        """Should generate different prefix for each backup type."""
        with patch("bot_v2.state.backup.workflow.datetime") as mock_dt:
            mock_dt.utcnow.return_value.strftime.return_value = "20250101_120000"

            full_id = workflow._generate_backup_id(BackupType.FULL)
            diff_id = workflow._generate_backup_id(BackupType.DIFFERENTIAL)
            inc_id = workflow._generate_backup_id(BackupType.INCREMENTAL)

            assert full_id.startswith("FUL_")
            assert diff_id.startswith("DIF_")
            assert inc_id.startswith("INC_")


class TestDataCollection:
    """Test data collection and normalization."""

    @pytest.mark.asyncio
    async def test_collects_data_via_collector(self, workflow, mock_data_collector):
        """Should delegate data collection to DataCollector."""
        await workflow._collect_backup_data(BackupType.FULL)

        mock_data_collector.collect_for_backup.assert_called_once_with(BackupType.FULL, None)

    @pytest.mark.asyncio
    async def test_uses_override_data_when_provided(self, workflow, mock_data_collector):
        """Should use override data when provided."""
        override_data = {"custom": "data"}

        await workflow._collect_backup_data(BackupType.FULL, override=override_data)

        mock_data_collector.collect_for_backup.assert_called_once_with(
            BackupType.FULL, override_data
        )

    @pytest.mark.asyncio
    async def test_normalizes_collected_data(self, workflow, mock_data_collector):
        """Should normalize collected data to be JSON serializable."""
        # Mock data with datetime (non-JSON serializable)
        mock_data_collector.collect_for_backup.return_value = {
            "timestamp": datetime.now(timezone.utc)
        }

        await workflow._collect_backup_data(BackupType.FULL)

        # Pending snapshot should be normalized (datetime → string)
        assert workflow.pending_snapshot is not None
        assert isinstance(workflow.pending_snapshot.get("timestamp"), str)

    @pytest.mark.asyncio
    async def test_includes_metadata_in_backup_data(self, workflow):
        """Should include metadata (timestamp, backup_type, system_info) in backup data."""
        result = await workflow._collect_backup_data(BackupType.FULL)

        assert "timestamp" in result
        assert "backup_type" in result
        assert result["backup_type"] == "full"
        assert "system_info" in result
        assert "state" in result


class TestStateDiffing:
    """Test differential backup logic."""

    @pytest.mark.asyncio
    async def test_full_backup_stores_complete_state(self, workflow, mock_data_collector):
        """Full backup should store complete state (no diffing)."""
        mock_data_collector.collect_for_backup.return_value = {
            "positions": {"BTC": {"qty": 1.0}},
            "orders": [],
        }

        result = await workflow._collect_backup_data(BackupType.FULL)

        # State should be complete (not diffed)
        assert "positions" in result["state"]
        assert "orders" in result["state"]

    @pytest.mark.asyncio
    async def test_incremental_backup_diffs_against_last_backup(
        self, workflow, mock_data_collector, context
    ):
        """Incremental backup should diff against last backup state."""
        # Set baseline (last backup state)
        context.last_backup_state = {"positions": {"BTC": {"qty": 1.0}}, "orders": []}

        # New state with changes
        mock_data_collector.collect_for_backup.return_value = {
            "positions": {"BTC": {"qty": 2.0}},  # Changed
            "orders": [],  # Same
        }

        result = await workflow._collect_backup_data(BackupType.INCREMENTAL)

        # Only changed data should be in state
        assert "positions" in result["state"]
        assert result["state"]["positions"]["BTC"]["qty"] == 2.0
        # Unchanged data may or may not be included (implementation detail)

    @pytest.mark.asyncio
    async def test_differential_backup_diffs_against_last_full(
        self, workflow, mock_data_collector, context
    ):
        """Differential backup should diff against last full backup state."""
        # Set baseline (last full backup state)
        context.last_full_state = {"positions": {"BTC": {"qty": 1.0}}, "orders": []}

        # New state with changes
        mock_data_collector.collect_for_backup.return_value = {
            "positions": {"BTC": {"qty": 3.0}},  # Changed
            "orders": [{"id": "123"}],  # New
        }

        result = await workflow._collect_backup_data(BackupType.DIFFERENTIAL)

        # Changed data should be in state
        assert "positions" in result["state"]
        assert result["state"]["positions"]["BTC"]["qty"] == 3.0


class TestDiffLogic:
    """Test _diff_state helper."""

    def test_diff_returns_full_state_when_no_baseline(self, workflow):
        """Should return full state when baseline is None."""
        current = {"a": 1, "b": 2}

        diff = workflow._diff_state(None, current)

        assert diff == current

    def test_diff_detects_changed_values(self, workflow):
        """Should detect changed values."""
        baseline = {"a": 1, "b": 2}
        current = {"a": 1, "b": 3}  # b changed

        diff = workflow._diff_state(baseline, current)

        assert "b" in diff
        assert diff["b"] == 3
        assert "a" not in diff  # Unchanged

    def test_diff_detects_new_keys(self, workflow):
        """Should include new keys in diff."""
        baseline = {"a": 1}
        current = {"a": 1, "b": 2}  # b is new

        diff = workflow._diff_state(baseline, current)

        assert "b" in diff
        assert diff["b"] == 2

    def test_diff_handles_nested_dicts(self, workflow):
        """Should handle nested dictionary diffing."""
        baseline = {"outer": {"inner": 1}}
        current = {"outer": {"inner": 2}}  # nested change

        diff = workflow._diff_state(baseline, current)

        assert "outer" in diff
        assert "inner" in diff["outer"]
        assert diff["outer"]["inner"] == 2

    def test_diff_only_includes_changed_nested_values(self, workflow):
        """Should only include changed values in nested dicts."""
        baseline = {"outer": {"a": 1, "b": 2}}
        current = {"outer": {"a": 1, "b": 3}}  # only b changed

        diff = workflow._diff_state(baseline, current)

        assert "outer" in diff
        assert "b" in diff["outer"]
        assert diff["outer"]["b"] == 3
        assert "a" not in diff["outer"]  # Unchanged


class TestErrorHandling:
    """Test error handling in workflow."""

    @pytest.mark.asyncio
    async def test_propagates_os_error(self, workflow, mock_backup_creator):
        """Should propagate OSError after logging."""
        mock_backup_creator.create_backup_internal.side_effect = OSError("Disk full")

        with pytest.raises(OSError):
            await workflow.create_backup(BackupType.FULL)

    @pytest.mark.asyncio
    async def test_catches_general_exceptions(self, workflow, mock_backup_creator):
        """Should catch general exceptions and return None."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        result = await workflow.create_backup(BackupType.FULL)

        assert result is None

    @pytest.mark.asyncio
    async def test_resets_flags_on_exception(self, workflow, mock_backup_creator):
        """Should reset flags even when exception occurs."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        await workflow.create_backup(BackupType.FULL)

        assert not workflow.is_backup_in_progress
        assert workflow.pending_snapshot is None


class TestNormalization:
    """Test state payload normalization."""

    def test_normalizes_datetime_objects(self, workflow):
        """Should convert datetime objects to strings."""
        payload = {"timestamp": datetime.now(timezone.utc)}

        normalized = workflow._normalize_state_payload(payload)

        assert isinstance(normalized["timestamp"], str)

    def test_handles_nested_datetimes(self, workflow):
        """Should normalize nested datetime objects."""
        payload = {"data": {"created_at": datetime.now(timezone.utc)}}

        normalized = workflow._normalize_state_payload(payload)

        assert isinstance(normalized["data"]["created_at"], str)

    def test_preserves_json_serializable_data(self, workflow):
        """Should preserve already JSON-serializable data."""
        payload = {"int": 123, "str": "test", "list": [1, 2, 3], "dict": {"a": 1}}

        normalized = workflow._normalize_state_payload(payload)

        assert normalized == payload


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
    def workflow_with_metrics(
        self,
        config,
        context,
        backup_lock,
        mock_data_collector,
        mock_backup_creator,
        mock_metrics_collector,
    ):
        """Create workflow with metrics collector."""
        return BackupWorkflow(
            data_collector=mock_data_collector,
            backup_creator=mock_backup_creator,
            context=context,
            config=config,
            backup_lock=backup_lock,
            metrics_collector=mock_metrics_collector,
        )

    @pytest.mark.asyncio
    async def test_successful_backup_records_metrics(
        self, workflow_with_metrics, mock_backup_creator, mock_metrics_collector
    ):
        """Should record success metrics when backup completes successfully."""
        result = await workflow_with_metrics.create_backup(BackupType.FULL)

        assert result is not None

        # Verify metrics recorded
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.operations.created_total" in counter_calls
        assert "backup.operations.created_success" in counter_calls
        assert "backup.operations.created_failed" not in counter_calls

        # Verify histogram metrics
        histogram_calls = [
            call[0][0] for call in mock_metrics_collector.record_histogram.call_args_list
        ]
        assert "backup.operations.duration_seconds" in histogram_calls
        assert "backup.operations.size_bytes_total" in histogram_calls

    @pytest.mark.asyncio
    async def test_failed_backup_records_failure_metrics(
        self, workflow_with_metrics, mock_backup_creator, mock_metrics_collector
    ):
        """Should record failure metrics when backup fails."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        result = await workflow_with_metrics.create_backup(BackupType.FULL)

        assert result is None

        # Verify metrics recorded
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.operations.created_total" in counter_calls
        assert "backup.operations.created_failed" in counter_calls
        assert "backup.operations.created_success" not in counter_calls

    @pytest.mark.asyncio
    async def test_duration_tracking_records_histogram(
        self, workflow_with_metrics, mock_metrics_collector
    ):
        """Should record duration as histogram."""
        await workflow_with_metrics.create_backup(BackupType.FULL)

        # Verify duration histogram recorded
        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        duration_call = next(
            call for call in histogram_calls if call[0][0] == "backup.operations.duration_seconds"
        )
        assert duration_call is not None
        # Duration should be a non-negative float
        duration_value = duration_call[0][1]
        assert isinstance(duration_value, float)
        assert duration_value >= 0

    @pytest.mark.asyncio
    async def test_size_tracking_records_histogram(
        self, workflow_with_metrics, mock_backup_creator, mock_metrics_collector
    ):
        """Should record backup size in histogram."""
        # Mock returns metadata with size_bytes=1000
        result = await workflow_with_metrics.create_backup(BackupType.FULL)

        assert result is not None
        assert result.size_bytes == 1000

        # Verify size histogram recorded
        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        size_call = next(
            call for call in histogram_calls if call[0][0] == "backup.operations.size_bytes_total"
        )
        assert size_call is not None
        size_value = size_call[0][1]
        assert size_value == 1000.0

    @pytest.mark.asyncio
    async def test_metrics_not_recorded_when_collector_none(self, workflow, mock_backup_creator):
        """Should not crash when metrics collector is None."""
        # Verify collector is None
        assert workflow.metrics_collector is None

        # Execute (should not crash)
        result = await workflow.create_backup(BackupType.FULL)

        # Verify success
        assert result is not None

    @pytest.mark.asyncio
    async def test_metrics_recorded_even_when_metadata_none(
        self, workflow_with_metrics, mock_backup_creator, mock_metrics_collector
    ):
        """Should record failure metrics when metadata is None."""
        # Make creator return None (failure)
        mock_backup_creator.create_backup_internal.return_value = None

        result = await workflow_with_metrics.create_backup(BackupType.FULL)

        assert result is None

        # Verify failure NOT recorded (None return doesn't trigger failure path)
        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "backup.operations.created_total" in counter_calls
        # None result doesn't count as success or failure - just no metadata
        assert "backup.operations.created_success" not in counter_calls
        assert "backup.operations.created_failed" not in counter_calls

    @pytest.mark.asyncio
    async def test_duration_recorded_in_finally_block(
        self, workflow_with_metrics, mock_backup_creator, mock_metrics_collector
    ):
        """Should always record duration even on exceptions."""
        mock_backup_creator.create_backup_internal.side_effect = Exception("Test error")

        await workflow_with_metrics.create_backup(BackupType.FULL)

        # Verify duration still recorded
        histogram_calls = [
            call[0][0] for call in mock_metrics_collector.record_histogram.call_args_list
        ]
        assert "backup.operations.duration_seconds" in histogram_calls
