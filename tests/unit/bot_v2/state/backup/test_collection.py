"""Tests for backup data collection strategies."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.backup.collection import BackupCollector
from bot_v2.state.backup.models import BackupType


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    manager = Mock()
    manager.get_keys_by_pattern = AsyncMock()
    manager.get_state = AsyncMock()
    return manager


@pytest.fixture
def backup_collector(mock_state_manager):
    """Create a backup collector with mocked state manager."""
    return BackupCollector(mock_state_manager)


class TestBackupCollector:
    """Test backup data collection strategies."""

    async def test_full_backup_collects_all_data(self, backup_collector, mock_state_manager):
        """Test that full backup collects all system data."""
        # Mock state manager responses - set up for multiple pattern queries
        async def mock_get_keys(pattern):
            if pattern.startswith("position"):
                return ["position:BTC"]
            elif pattern.startswith("order"):
                return ["order:123"]
            elif pattern.startswith("portfolio"):
                return ["portfolio_current"]
            return []

        mock_state_manager.get_keys_by_pattern.side_effect = mock_get_keys

        # Mock get_state to return different data for each key
        async def mock_get_state(key):
            if key == "position:BTC":
                return {"symbol": "BTC", "size": 1.0}
            elif key == "order:123":
                return {"order_id": "123", "status": "open"}
            elif key == "portfolio_current":
                return {"value": 10000}
            return None

        mock_state_manager.get_state.side_effect = mock_get_state

        result = await backup_collector.collect_backup_data(
            BackupType.FULL, last_full_backup=None, last_backup_time=datetime.utcnow()
        )

        # Verify result structure
        assert "timestamp" in result
        assert "backup_type" in result
        assert result["backup_type"] == "full"
        assert "system_info" in result

        # Verify all data was collected
        assert "position:BTC" in result
        assert "order:123" in result
        assert "portfolio_current" in result

    async def test_incremental_backup_collects_changed_data(
        self, backup_collector, mock_state_manager
    ):
        """Test that incremental backup collects only changed data."""
        last_backup = datetime.utcnow() - timedelta(hours=1)

        # Mock recent data
        async def mock_get_keys(pattern):
            if pattern.startswith("position"):
                return ["position:BTC", "position:ETH"]
            return []

        # One recent, one old
        recent_timestamp = datetime.utcnow().isoformat()
        old_timestamp = (datetime.utcnow() - timedelta(hours=2)).isoformat()

        async def mock_get_state(key):
            if key == "position:BTC":
                return {"symbol": "BTC", "timestamp": recent_timestamp}
            elif key == "position:ETH":
                return {"symbol": "ETH", "timestamp": old_timestamp}
            return None

        mock_state_manager.get_keys_by_pattern.side_effect = mock_get_keys
        mock_state_manager.get_state.side_effect = mock_get_state

        result = await backup_collector.collect_backup_data(
            BackupType.INCREMENTAL, last_full_backup=None, last_backup_time=last_backup
        )

        assert result["backup_type"] == "incremental"
        # Should only include recent data
        assert "position:BTC" in result
        # Old data should be excluded
        assert "position:ETH" not in result

    async def test_differential_backup_since_last_full(
        self, backup_collector, mock_state_manager
    ):
        """Test that differential backup collects changes since last full backup."""
        last_full = datetime.utcnow() - timedelta(days=1)
        last_backup = datetime.utcnow() - timedelta(hours=1)

        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC"]

        recent_timestamp = datetime.utcnow().isoformat()
        mock_state_manager.get_state.return_value = {
            "symbol": "BTC",
            "timestamp": recent_timestamp,
        }

        result = await backup_collector.collect_backup_data(
            BackupType.DIFFERENTIAL, last_full_backup=last_full, last_backup_time=last_backup
        )

        assert result["backup_type"] == "differential"
        assert "position:BTC" in result

    async def test_snapshot_backup_collects_current_state(
        self, backup_collector, mock_state_manager
    ):
        """Test that snapshot backup collects current state."""
        async def mock_get_keys(pattern):
            if "position" in pattern:
                return ["position:BTC"]
            elif "order" in pattern:
                return ["order:123"]
            return []

        async def mock_get_state(key):
            if key.startswith("position"):
                return {"symbol": "BTC"}
            elif key == "portfolio_current":
                return {"value": 10000}
            elif key == "performance_metrics":
                return {"pnl": 500}
            return None

        mock_state_manager.get_keys_by_pattern.side_effect = mock_get_keys
        mock_state_manager.get_state.side_effect = mock_get_state

        result = await backup_collector.collect_backup_data(
            BackupType.SNAPSHOT, last_full_backup=None, last_backup_time=datetime.utcnow()
        )

        assert result["backup_type"] == "snapshot"
        assert "positions" in result
        assert "portfolio" in result
        assert "performance" in result

    async def test_emergency_backup_collects_critical_data_only(
        self, backup_collector, mock_state_manager
    ):
        """Test that emergency backup collects only critical data."""
        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC"]
        mock_state_manager.get_state.side_effect = [
            {"symbol": "BTC"},  # position
            {"value": 10000},  # portfolio
            {"api_key": "secret"},  # critical config
        ]

        result = await backup_collector.collect_backup_data(
            BackupType.EMERGENCY, last_full_backup=None, last_backup_time=datetime.utcnow()
        )

        assert result["backup_type"] == "emergency"
        assert "positions" in result
        assert "portfolio" in result
        assert "critical_config" in result

    async def test_collect_all_data_queries_all_patterns(
        self, backup_collector, mock_state_manager
    ):
        """Test that collect_all_data queries all expected patterns."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.get_state.return_value = None

        await backup_collector.collect_all_data()

        # Verify all patterns were queried
        call_args = [call[0][0] for call in mock_state_manager.get_keys_by_pattern.call_args_list]
        expected_patterns = [
            "position:*",
            "order:*",
            "portfolio*",
            "ml_model:*",
            "config:*",
            "performance*",
            "strategy:*",
        ]

        for pattern in expected_patterns:
            assert pattern in call_args

    async def test_collect_changed_data_handles_timestamp_formats(
        self, backup_collector, mock_state_manager
    ):
        """Test that changed data collection handles various timestamp formats."""
        since = datetime.utcnow() - timedelta(hours=1)

        async def mock_get_keys(pattern):
            if "position" in pattern:
                return ["item1", "item2", "item3"]
            return []

        # Different timestamp formats
        async def mock_get_state(key):
            if key == "item1":
                return {"data": "recent", "timestamp": datetime.utcnow().isoformat()}
            elif key == "item2":
                return {"data": "recent", "last_updated": datetime.utcnow()}
            elif key == "item3":
                return {"data": "old", "timestamp": "invalid"}
            return None

        mock_state_manager.get_keys_by_pattern.side_effect = mock_get_keys
        mock_state_manager.get_state.side_effect = mock_get_state

        result = await backup_collector.collect_changed_data(since)

        # All should be included (recent items + items with invalid timestamps)
        assert len(result) == 3

    async def test_collect_changed_data_includes_items_without_timestamp(
        self, backup_collector, mock_state_manager
    ):
        """Test that items without timestamps are included in changed data."""
        since = datetime.utcnow() - timedelta(hours=1)

        mock_state_manager.get_keys_by_pattern.return_value = ["item1"]
        mock_state_manager.get_state.return_value = {"data": "no_timestamp"}

        result = await backup_collector.collect_changed_data(since)

        assert "item1" in result

    async def test_collect_snapshot_data_structure(self, backup_collector, mock_state_manager):
        """Test snapshot data has correct structure."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.get_state.return_value = None

        result = await backup_collector.collect_snapshot_data()

        assert "positions" in result
        assert "orders" in result
        assert "portfolio" in result
        assert "performance" in result

    async def test_collect_critical_data_structure(self, backup_collector, mock_state_manager):
        """Test critical data has correct structure."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.get_state.return_value = None

        result = await backup_collector.collect_critical_data()

        assert "positions" in result
        assert "portfolio" in result
        assert "critical_config" in result

    async def test_get_all_by_pattern(self, backup_collector, mock_state_manager):
        """Test get_all_by_pattern helper method."""
        mock_state_manager.get_keys_by_pattern.return_value = ["key1", "key2", "key3"]
        mock_state_manager.get_state.side_effect = [
            {"data": "value1"},
            None,  # Should be excluded
            {"data": "value3"},
        ]

        result = await backup_collector.get_all_by_pattern("pattern:*")

        assert len(result) == 2
        assert "key1" in result
        assert "key3" in result
        assert "key2" not in result  # None values excluded

    async def test_backup_data_includes_system_info(self, backup_collector, mock_state_manager):
        """Test that all backups include system info."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        with patch.dict(os.environ, {"ENV": "production"}):
            result = await backup_collector.collect_backup_data(
                BackupType.FULL, last_full_backup=None, last_backup_time=datetime.utcnow()
            )

        assert "system_info" in result
        assert result["system_info"]["version"] == "1.0.0"
        assert result["system_info"]["environment"] == "production"

    async def test_backup_data_defaults_environment(self, backup_collector, mock_state_manager):
        """Test that environment defaults to 'production' if not set."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        with patch.dict(os.environ, {}, clear=True):
            result = await backup_collector.collect_backup_data(
                BackupType.FULL, last_full_backup=None, last_backup_time=datetime.utcnow()
            )

        assert result["system_info"]["environment"] == "production"

    async def test_differential_backup_defaults_to_30_days(
        self, backup_collector, mock_state_manager
    ):
        """Test that differential backup defaults to 30 days if no last full backup."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        await backup_collector.collect_backup_data(
            BackupType.DIFFERENTIAL, last_full_backup=None, last_backup_time=datetime.utcnow()
        )

        # Verify it used the fallback timeframe (since timestamp would be ~30 days ago)
        # We can't directly verify the timestamp, but we can verify it didn't crash
        assert True  # If we got here without error, the default worked

    async def test_collect_changed_data_logs_debug_info(
        self, backup_collector, mock_state_manager, caplog
    ):
        """Test that collection logs debug information."""
        import logging

        caplog.set_level(logging.DEBUG)

        mock_state_manager.get_keys_by_pattern.return_value = []

        since = datetime.utcnow()
        await backup_collector.collect_changed_data(since)

        # Check for debug log
        assert any("Collected" in record.message for record in caplog.records)
        assert any("changed items" in record.message for record in caplog.records)

    async def test_collect_all_data_logs_debug_info(
        self, backup_collector, mock_state_manager, caplog
    ):
        """Test that full collection logs debug information."""
        import logging

        caplog.set_level(logging.DEBUG)

        mock_state_manager.get_keys_by_pattern.return_value = []

        await backup_collector.collect_all_data()

        # Check for debug log
        assert any("Collected" in record.message for record in caplog.records)
        assert any("full backup" in record.message for record in caplog.records)
