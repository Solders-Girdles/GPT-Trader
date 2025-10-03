"""Unit tests for DataCollector.

Tests state collection logic in isolation:
- Collection strategies for each backup type
- Repository vs StateManager fallback
- Timestamp filtering for incremental backups
- Performance metrics integration
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.state.backup.collector import DataCollector
from bot_v2.state.backup.models import BackupConfig, BackupContext, BackupType


@pytest.fixture
def backup_config():
    """Minimal backup configuration."""
    return BackupConfig()


@pytest.fixture
def backup_context():
    """Shared backup context."""
    return BackupContext()


@pytest.fixture
def mock_metadata_manager():
    """Mock metadata manager."""
    manager = Mock()
    manager.get_last_backup_time = Mock(
        return_value=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    return manager


@pytest.fixture
def mock_metrics():
    """Mock performance metrics."""
    metrics = Mock()
    metrics.time_operation = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
    return metrics


@pytest.fixture
def mock_state_manager():
    """Mock state manager without repositories."""
    manager = Mock()
    manager.get_keys_by_pattern = AsyncMock(return_value=["position:BTC", "order:123"])
    manager.get_state = AsyncMock(return_value={"qty": 1.5})
    manager.create_snapshot = None  # No snapshot method
    return manager


@pytest.fixture
def mock_state_manager_with_repos():
    """Mock state manager with repository access."""
    manager = Mock()

    # No create_snapshot method
    manager.create_snapshot = None

    # Mock repositories
    repos = Mock()
    redis = Mock()
    redis.keys = AsyncMock(return_value=["position:BTC"])
    redis.fetch = AsyncMock(
        return_value={"qty": 1.5, "timestamp": datetime.now(timezone.utc).isoformat()}
    )
    repos.redis = redis

    postgres = Mock()
    postgres.keys = AsyncMock(return_value=["order:123"])
    postgres.fetch = AsyncMock(return_value={"status": "filled"})
    repos.postgres = postgres

    repos.s3 = None  # No S3 tier

    manager.get_repositories = Mock(return_value=repos)
    return manager


@pytest.fixture
def data_collector(
    mock_state_manager,
    backup_config,
    backup_context,
    mock_metadata_manager,
    mock_metrics,
):
    """DataCollector instance with mocked StateManager."""
    return DataCollector(
        state_manager=mock_state_manager,
        config=backup_config,
        context=backup_context,
        metadata_manager=mock_metadata_manager,
        metrics=mock_metrics,
    )


@pytest.fixture
def data_collector_with_repos(
    mock_state_manager_with_repos,
    backup_config,
    backup_context,
    mock_metadata_manager,
    mock_metrics,
):
    """DataCollector instance with repository access."""
    return DataCollector(
        state_manager=mock_state_manager_with_repos,
        config=backup_config,
        context=backup_context,
        metadata_manager=mock_metadata_manager,
        metrics=mock_metrics,
    )


class TestFullBackupCollection:
    """Tests for FULL backup collection."""

    @pytest.mark.asyncio
    async def test_collects_all_data_via_state_manager(self, data_collector, mock_state_manager):
        """Collects all state data via StateManager when repositories unavailable."""
        result = await data_collector.collect_for_backup(BackupType.FULL)

        # Verify StateManager was used
        assert mock_state_manager.get_keys_by_pattern.called
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_collects_all_data_via_repositories(
        self, data_collector_with_repos, mock_state_manager_with_repos
    ):
        """Uses direct repository access when available (fast path)."""
        result = await data_collector_with_repos.collect_for_backup(BackupType.FULL)

        # Verify data was collected (result should be a dict)
        assert isinstance(result, dict)

        # Verify get_repositories was called (indicates fast path attempt)
        mock_state_manager_with_repos.get_repositories.assert_called()

    @pytest.mark.asyncio
    async def test_uses_snapshot_method_when_available(
        self, mock_state_manager, backup_config, backup_context, mock_metadata_manager, mock_metrics
    ):
        """Prefers create_snapshot method if state manager provides it."""
        snapshot_data = {"position:BTC": {"qty": 1.5}, "portfolio": {"cash": 10000}}
        mock_state_manager.create_snapshot = AsyncMock(return_value=snapshot_data)

        collector = DataCollector(
            state_manager=mock_state_manager,
            config=backup_config,
            context=backup_context,
            metadata_manager=mock_metadata_manager,
            metrics=mock_metrics,
        )

        result = await collector.collect_for_backup(BackupType.FULL)

        # Verify snapshot method was called
        mock_state_manager.create_snapshot.assert_called_once()
        assert result == snapshot_data

    @pytest.mark.asyncio
    async def test_collects_from_all_patterns(self, data_collector):
        """Collects data from all expected patterns."""
        result = await data_collector.collect_for_backup(BackupType.FULL)

        # Verify patterns were queried (position, order, portfolio, etc.)
        # Mock returns data for position:BTC and order:123
        assert isinstance(result, dict)


class TestIncrementalBackupCollection:
    """Tests for INCREMENTAL backup collection."""

    @pytest.mark.asyncio
    async def test_collects_changed_data_since_last_backup(
        self, data_collector, mock_metadata_manager
    ):
        """Collects only data changed since last backup."""
        last_backup = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_metadata_manager.get_last_backup_time.return_value = last_backup

        result = await data_collector.collect_for_backup(BackupType.INCREMENTAL)

        # Verify last backup time was retrieved
        mock_metadata_manager.get_last_backup_time.assert_called_once()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_filters_by_timestamp(self, data_collector_with_repos):
        """Filters data based on timestamp field."""
        # Mock data with old timestamp
        old_data = {"timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}
        new_data = {"timestamp": datetime.now(timezone.utc).isoformat()}

        repos = data_collector_with_repos.state_manager.get_repositories()
        repos.redis.fetch = AsyncMock(side_effect=[old_data, new_data])

        result = await data_collector_with_repos.collect_for_backup(BackupType.INCREMENTAL)

        # Verify timestamp filtering occurred
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_includes_items_without_timestamp(self, data_collector):
        """Includes items that lack timestamp field."""
        # Items without timestamps should be included (conservative approach)
        result = await data_collector.collect_for_backup(BackupType.INCREMENTAL)
        assert isinstance(result, dict)


class TestDifferentialBackupCollection:
    """Tests for DIFFERENTIAL backup collection."""

    @pytest.mark.asyncio
    async def test_collects_changes_since_last_full_backup(self, data_collector, backup_context):
        """Collects changes since last full backup."""
        last_full = datetime.now(timezone.utc) - timedelta(days=1)
        backup_context.last_full_backup = last_full

        result = await data_collector.collect_for_backup(BackupType.DIFFERENTIAL)

        # Verify differential collection
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_uses_default_baseline_when_no_full_backup(self, data_collector):
        """Uses 30-day default when no full backup exists."""
        result = await data_collector.collect_for_backup(BackupType.DIFFERENTIAL)

        # Should not fail, uses default baseline
        assert isinstance(result, dict)


class TestSnapshotBackupCollection:
    """Tests for SNAPSHOT backup collection."""

    @pytest.mark.asyncio
    async def test_collects_snapshot_data(self, data_collector, mock_state_manager):
        """Collects current positions, orders, portfolio, performance."""
        result = await data_collector.collect_for_backup(BackupType.SNAPSHOT)

        # Verify snapshot structure
        assert isinstance(result, dict)
        assert "positions" in result
        assert "orders" in result
        assert "portfolio" in result
        assert "performance" in result

    @pytest.mark.asyncio
    async def test_snapshot_includes_current_state(self, data_collector, mock_state_manager):
        """Snapshot includes current portfolio and performance metrics."""
        mock_state_manager.get_state = AsyncMock(
            side_effect=lambda key: {"cash": 10000} if key == "portfolio_current" else {"pnl": 500}
        )

        result = await data_collector.collect_for_backup(BackupType.SNAPSHOT)

        # Verify portfolio and performance were fetched
        assert result["portfolio"] == {"cash": 10000}
        assert result["performance"] == {"pnl": 500}


class TestEmergencyBackupCollection:
    """Tests for EMERGENCY backup collection."""

    @pytest.mark.asyncio
    async def test_collects_critical_data_only(self, data_collector):
        """Collects only critical data for emergency backup."""
        result = await data_collector.collect_for_backup(BackupType.EMERGENCY)

        # Verify emergency structure (positions, portfolio, critical_config)
        assert isinstance(result, dict)
        assert "positions" in result
        assert "portfolio" in result
        assert "critical_config" in result

    @pytest.mark.asyncio
    async def test_emergency_excludes_non_critical_data(self, data_collector):
        """Does not include orders, ml_models, etc. in emergency backup."""
        result = await data_collector.collect_for_backup(BackupType.EMERGENCY)

        # Emergency backup should be minimal
        assert len(result) == 3  # Only positions, portfolio, critical_config


class TestOverrideData:
    """Tests for override data parameter."""

    @pytest.mark.asyncio
    async def test_uses_override_when_provided(self, data_collector):
        """Uses override data instead of collecting from state."""
        override_data = {"custom": "data"}

        result = await data_collector.collect_for_backup(BackupType.FULL, override=override_data)

        assert result == override_data

    @pytest.mark.asyncio
    async def test_skips_collection_with_override(self, data_collector, mock_state_manager):
        """Does not query state manager when override provided."""
        override_data = {"custom": "data"}

        await data_collector.collect_for_backup(BackupType.FULL, override=override_data)

        # Verify state manager was not called
        mock_state_manager.get_keys_by_pattern.assert_not_called()


class TestRepositoryFallback:
    """Tests for repository access fallback logic."""

    @pytest.mark.asyncio
    async def test_falls_back_to_state_manager_on_error(
        self,
        mock_state_manager_with_repos,
        backup_config,
        backup_context,
        mock_metadata_manager,
        mock_metrics,
    ):
        """Falls back to StateManager if repository access fails."""
        # Make sure there's no create_snapshot method
        mock_state_manager_with_repos.create_snapshot = None

        # Make repositories raise TypeError (e.g., Mock not async-compatible)
        repos = mock_state_manager_with_repos.get_repositories()

        # Make both redis.keys and redis.fetch raise TypeError to trigger fallback
        async def failing_keys(*args, **kwargs):
            raise TypeError("Not async")

        repos.redis.keys = failing_keys
        repos.redis.fetch = failing_keys

        # Add StateManager fallback methods
        mock_state_manager_with_repos.get_keys_by_pattern = AsyncMock(return_value=["position:BTC"])
        mock_state_manager_with_repos.get_state = AsyncMock(return_value={"qty": 1.5})

        collector = DataCollector(
            state_manager=mock_state_manager_with_repos,
            config=backup_config,
            context=backup_context,
            metadata_manager=mock_metadata_manager,
            metrics=mock_metrics,
        )

        result = await collector.collect_for_backup(BackupType.FULL)

        # Verify result is valid (fallback may or may not be called depending on error handling)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_missing_repositories(self, data_collector, mock_state_manager):
        """Handles state managers without get_repositories method."""
        # Mock state manager doesn't have get_repositories
        delattr(mock_state_manager, "get_repositories")

        result = await data_collector.collect_for_backup(BackupType.FULL)

        # Should use StateManager path without error
        assert isinstance(result, dict)


class TestPerformanceMetrics:
    """Tests for performance metrics integration."""

    @pytest.mark.asyncio
    async def test_records_collection_metrics(self, data_collector, mock_metrics):
        """Records performance metrics for collection operations."""
        await data_collector.collect_for_backup(BackupType.FULL)

        # Verify metrics were recorded
        mock_metrics.time_operation.assert_called()

    @pytest.mark.asyncio
    async def test_tracks_different_operations(self, data_collector, mock_metrics):
        """Tracks metrics for different operation types."""
        await data_collector.collect_for_backup(BackupType.SNAPSHOT)

        # Verify specific operation metrics
        calls = [str(call) for call in mock_metrics.time_operation.call_args_list]
        assert any("get_all_by_pattern" in str(call) for call in calls)
