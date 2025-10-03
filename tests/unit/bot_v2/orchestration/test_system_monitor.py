"""
Comprehensive tests for SystemMonitor.

Covers metrics publishing, config refresh, telemetry, and position reconciliation.
"""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.configuration import ConfigValidationError, Profile
from bot_v2.orchestration.system_monitor import SystemMonitor


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot."""
    bot = Mock()
    bot.bot_id = "test-bot-123"
    bot.running = True
    bot.start_time = datetime.now(UTC)
    bot.last_decisions = {}
    bot.order_stats = {}
    bot._last_positions = {}

    # Mock config
    bot.config = Mock()
    bot.config.profile = Profile.DEV

    # Mock broker
    bot.broker = Mock()
    bot.broker.list_positions = Mock(return_value=[])
    bot.broker.list_balances = Mock(return_value=[])

    # Mock stores
    bot.orders_store = Mock()
    bot.orders_store.get_open_orders = Mock(return_value=[])

    bot.event_store = Mock()
    bot.event_store.append_metric = Mock()

    # Mock config controller
    bot.config_controller = None

    # Mock apply_config_change
    bot.apply_config_change = Mock()

    return bot


@pytest.fixture
def mock_telemetry():
    """Create mock AccountTelemetryService."""
    telemetry = Mock()
    telemetry.latest_snapshot = {"account_id": "test-123", "balance": 10000}
    return telemetry


@pytest.fixture
def mock_position():
    """Create mock position."""
    position = Mock()
    position.symbol = "BTC-USD"
    position.quantity = Decimal("0.5")
    position.side = "LONG"
    position.entry_price = Decimal("50000")
    position.mark_price = Decimal("51000")
    return position


@pytest.fixture
def mock_balance():
    """Create mock balance."""
    balance = Mock()
    balance.asset = "USD"
    balance.available = Decimal("10000")
    return balance


class TestSystemMonitorInitialization:
    """Test SystemMonitor initialization."""

    def test_init_without_telemetry(self, mock_bot):
        """Should initialize without telemetry service."""
        monitor = SystemMonitor(mock_bot)

        assert monitor._bot is mock_bot
        assert monitor._account_telemetry is None

    def test_init_with_telemetry(self, mock_bot, mock_telemetry):
        """Should initialize with telemetry service."""
        monitor = SystemMonitor(mock_bot, account_telemetry=mock_telemetry)

        assert monitor._account_telemetry is mock_telemetry

    @patch("bot_v2.orchestration.system_monitor.ResourceCollector")
    def test_init_creates_resource_collector(self, mock_rc_class, mock_bot):
        """Should create ResourceCollector if available."""
        monitor = SystemMonitor(mock_bot)

        mock_rc_class.assert_called_once()
        assert monitor._resource_collector is not None

    @patch(
        "bot_v2.orchestration.system_monitor.ResourceCollector",
        side_effect=Exception("psutil unavailable"),
    )
    def test_init_handles_resource_collector_error(self, mock_rc_class, mock_bot):
        """Should gracefully handle ResourceCollector creation failure."""
        monitor = SystemMonitor(mock_bot)

        # Should not crash, just set to None
        assert monitor._resource_collector is None

    def test_attach_account_telemetry(self, mock_bot, mock_telemetry):
        """Should attach telemetry service."""
        monitor = SystemMonitor(mock_bot)
        monitor.attach_account_telemetry(mock_telemetry)

        assert monitor._account_telemetry is mock_telemetry


class TestLogStatus:
    """Test log_status method."""

    @pytest.mark.asyncio
    async def test_log_status_basic(self, mock_bot, mock_position, mock_balance):
        """Should log status and publish metrics."""
        mock_bot.broker.list_positions.return_value = [mock_position]
        mock_bot.broker.list_balances.return_value = [mock_balance]

        monitor = SystemMonitor(mock_bot)
        await monitor.log_status()

        # Should append metric
        mock_bot.event_store.append_metric.assert_called_once()
        call_args = mock_bot.event_store.append_metric.call_args
        assert call_args[1]["bot_id"] == "test-bot-123"
        assert call_args[1]["metrics"]["event_type"] == "cycle_metrics"

    @pytest.mark.asyncio
    async def test_log_status_with_positions(self, mock_bot, mock_position, mock_balance):
        """Should include positions in metrics."""
        mock_bot.broker.list_positions.return_value = [mock_position]
        mock_bot.broker.list_balances.return_value = [mock_balance]

        monitor = SystemMonitor(mock_bot)
        await monitor.log_status()

        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]

        assert len(metrics["positions"]) == 1
        assert metrics["positions"][0]["symbol"] == "BTC-USD"
        assert metrics["positions"][0]["quantity"] == 0.5
        assert metrics["equity"] == 10000

    @pytest.mark.asyncio
    async def test_log_status_broker_position_error(self, mock_bot):
        """Should handle broker position fetch errors."""
        mock_bot.broker.list_positions.side_effect = Exception("API error")
        mock_bot.broker.list_balances.return_value = []

        monitor = SystemMonitor(mock_bot)
        await monitor.log_status()

        # Should not crash, just log warning
        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]
        assert metrics["positions"] == []

    @pytest.mark.asyncio
    async def test_log_status_broker_balance_error(self, mock_bot):
        """Should handle broker balance fetch errors."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.side_effect = Exception("API error")

        monitor = SystemMonitor(mock_bot)
        await monitor.log_status()

        # Should default equity to 0
        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]
        assert metrics["equity"] == 0

    @pytest.mark.asyncio
    async def test_log_status_includes_telemetry_snapshot(self, mock_bot, mock_telemetry):
        """Should include telemetry snapshot in metrics."""
        monitor = SystemMonitor(mock_bot, account_telemetry=mock_telemetry)
        await monitor.log_status()

        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]

        assert "account_snapshot" in metrics
        assert metrics["account_snapshot"]["account_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_log_status_without_telemetry_snapshot(self, mock_bot):
        """Should work without telemetry snapshot."""
        monitor = SystemMonitor(mock_bot)
        await monitor.log_status()

        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]

        assert "account_snapshot" not in metrics

    @pytest.mark.asyncio
    @patch("bot_v2.orchestration.system_monitor.ResourceCollectorType")
    async def test_log_status_with_system_metrics(self, mock_rc_type, mock_bot):
        """Should include system metrics when ResourceCollector available."""
        # Create mock usage
        mock_usage = Mock()
        mock_usage.cpu_percent = 45.5
        mock_usage.memory_percent = 60.0
        mock_usage.memory_mb = 512.0
        mock_usage.disk_percent = 75.0
        mock_usage.disk_gb = 100.0
        mock_usage.network_sent_mb = 50.0
        mock_usage.network_recv_mb = 100.0
        mock_usage.open_files = 100
        mock_usage.threads = 20

        mock_collector = Mock()
        mock_collector.collect.return_value = mock_usage
        mock_rc_type.return_value = mock_collector

        monitor = SystemMonitor(mock_bot)
        monitor._resource_collector = mock_collector
        await monitor.log_status()

        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]

        assert "system" in metrics
        assert metrics["system"]["cpu_percent"] == 45.5
        assert metrics["system"]["memory_used_mb"] == 512.0

    @pytest.mark.asyncio
    async def test_log_status_resource_collector_error(self, mock_bot):
        """Should handle ResourceCollector errors gracefully."""
        mock_collector = Mock()
        mock_collector.collect.side_effect = Exception("Collection failed")

        monitor = SystemMonitor(mock_bot)
        monitor._resource_collector = mock_collector
        await monitor.log_status()

        # Should not crash
        call_args = mock_bot.event_store.append_metric.call_args
        metrics = call_args[1]["metrics"]
        assert "system" not in metrics


class TestPublishMetrics:
    """Test _publish_metrics method."""

    def test_publish_metrics_to_event_store(self, mock_bot):
        """Should publish metrics to event store."""
        monitor = SystemMonitor(mock_bot)
        metrics = {"event_type": "cycle_metrics", "equity": 10000}

        monitor._publish_metrics(metrics)

        mock_bot.event_store.append_metric.assert_called_once()

    def test_publish_metrics_event_store_error(self, mock_bot):
        """Should handle event store errors."""
        mock_bot.event_store.append_metric.side_effect = Exception("Store error")

        monitor = SystemMonitor(mock_bot)
        metrics = {"event_type": "cycle_metrics"}

        # Should not crash
        monitor._publish_metrics(metrics)

    def test_publish_metrics_writes_json_file(self, mock_bot, tmp_path):
        """Should write metrics to JSON file."""
        with patch("bot_v2.orchestration.system_monitor.RUNTIME_DATA_DIR", tmp_path):
            monitor = SystemMonitor(mock_bot)
            metrics = {"event_type": "cycle_metrics", "equity": 10000}

            monitor._publish_metrics(metrics)

            # Check file was written
            metrics_file = tmp_path / "perps_bot" / "dev" / "metrics.json"
            assert metrics_file.exists()

            with open(metrics_file) as f:
                saved_metrics = json.load(f)

            assert saved_metrics["equity"] == 10000

    def test_publish_metrics_file_write_error(self, mock_bot):
        """Should handle file write errors."""
        monitor = SystemMonitor(mock_bot)
        metrics = {"event_type": "cycle_metrics"}

        with patch("bot_v2.orchestration.system_monitor.RUNTIME_DATA_DIR", Path("/invalid/path")):
            # Should not crash
            monitor._publish_metrics(metrics)

    @patch("bot_v2.orchestration.system_monitor._get_plog")
    def test_publish_metrics_plog_error(self, mock_plog, mock_bot):
        """Should handle plog errors."""
        mock_plog.return_value.log_event.side_effect = Exception("Plog error")

        monitor = SystemMonitor(mock_bot)
        metrics = {"event_type": "cycle_metrics"}

        # Should not crash
        monitor._publish_metrics(metrics)


class TestCheckConfigUpdates:
    """Test check_config_updates method."""

    def test_check_config_updates_no_controller(self, mock_bot):
        """Should handle missing config controller."""
        monitor = SystemMonitor(mock_bot)

        # Should not crash
        monitor.check_config_updates()

    def test_check_config_updates_no_changes(self, mock_bot):
        """Should handle no configuration changes."""
        mock_controller = Mock()
        mock_controller.refresh_if_changed.return_value = None
        mock_bot.config_controller = mock_controller

        monitor = SystemMonitor(mock_bot)
        monitor.check_config_updates()

        # Should not apply changes
        mock_bot.apply_config_change.assert_not_called()

    def test_check_config_updates_with_diff(self, mock_bot):
        """Should apply configuration changes with diff."""
        mock_change = Mock(spec=ConfigChange)
        mock_change.diff = {"update_interval": {"old": 60, "new": 30}}

        mock_controller = Mock()
        mock_controller.refresh_if_changed.return_value = mock_change
        mock_controller.current = Mock()
        mock_controller.current.profile = Profile.DEV
        mock_bot.config_controller = mock_controller

        monitor = SystemMonitor(mock_bot)
        monitor.check_config_updates()

        # Should apply change
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_without_diff(self, mock_bot):
        """Should handle config changes without diff."""
        mock_change = Mock(spec=ConfigChange)
        mock_change.diff = None

        mock_controller = Mock()
        mock_controller.refresh_if_changed.return_value = mock_change
        mock_controller.current = Mock()
        mock_controller.current.profile = Profile.DEV
        mock_bot.config_controller = mock_controller

        monitor = SystemMonitor(mock_bot)
        monitor.check_config_updates()

        # Should still apply
        mock_bot.apply_config_change.assert_called_once()

    def test_check_config_updates_validation_error(self, mock_bot):
        """Should handle configuration validation errors."""
        mock_controller = Mock()
        mock_controller.refresh_if_changed.side_effect = ConfigValidationError("Invalid config")
        mock_bot.config_controller = mock_controller

        monitor = SystemMonitor(mock_bot)
        monitor.check_config_updates()

        # Should not apply changes
        mock_bot.apply_config_change.assert_not_called()

    def test_check_config_updates_apply_error(self, mock_bot):
        """Should handle errors when applying config changes."""
        mock_change = Mock(spec=ConfigChange)
        mock_change.diff = {"key": "value"}

        mock_controller = Mock()
        mock_controller.refresh_if_changed.return_value = mock_change
        mock_controller.current = Mock()
        mock_controller.current.profile = Profile.DEV
        mock_bot.config_controller = mock_controller

        mock_bot.apply_config_change.side_effect = Exception("Apply failed")

        monitor = SystemMonitor(mock_bot)
        monitor.check_config_updates()

        # Should still consume pending change
        mock_controller.consume_pending_change.assert_called_once()


class TestWriteHealthStatus:
    """Test write_health_status method."""

    def test_write_health_status_ok(self, mock_bot, tmp_path):
        """Should write OK health status."""
        with patch("bot_v2.orchestration.system_monitor.RUNTIME_DATA_DIR", tmp_path):
            monitor = SystemMonitor(mock_bot)
            monitor.write_health_status(ok=True, message="All systems operational")

            status_file = tmp_path / "perps_bot" / "dev" / "health.json"
            assert status_file.exists()

            with open(status_file) as f:
                status = json.load(f)

            assert status["ok"] is True
            assert status["message"] == "All systems operational"
            assert "timestamp" in status

    def test_write_health_status_error(self, mock_bot, tmp_path):
        """Should write error health status."""
        with patch("bot_v2.orchestration.system_monitor.RUNTIME_DATA_DIR", tmp_path):
            monitor = SystemMonitor(mock_bot)
            monitor.write_health_status(
                ok=False, message="System degraded", error="Database connection lost"
            )

            status_file = tmp_path / "perps_bot" / "dev" / "health.json"
            with open(status_file) as f:
                status = json.load(f)

            assert status["ok"] is False
            assert status["error"] == "Database connection lost"


class TestPositionReconciliation:
    """Test position reconciliation loop."""

    @pytest.mark.asyncio
    async def test_position_reconciliation_initial_positions(self, mock_bot, mock_position):
        """Should set initial positions."""
        mock_bot.broker.list_positions.return_value = [mock_position]

        monitor = SystemMonitor(mock_bot)

        # Run one iteration
        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should set _last_positions
        assert "BTC-USD" in mock_bot._last_positions

    @pytest.mark.asyncio
    async def test_position_reconciliation_detects_changes(self, mock_bot):
        """Should detect position changes."""
        # Set initial position
        mock_bot._last_positions = {"BTC-USD": {"quantity": "0.5", "side": "LONG"}}

        # New position with different quantity
        new_position = Mock()
        new_position.symbol = "BTC-USD"
        new_position.quantity = Decimal("1.0")
        new_position.side = "LONG"

        mock_bot.broker.list_positions.return_value = [new_position]

        monitor = SystemMonitor(mock_bot)

        # Run one iteration
        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should log position change
        mock_bot.event_store.append_metric.assert_called()
        call_args = mock_bot.event_store.append_metric.call_args
        assert call_args[1]["metrics"]["event_type"] == "position_drift"

    @pytest.mark.asyncio
    async def test_position_reconciliation_detects_closed_positions(self, mock_bot):
        """Should detect closed positions."""
        # Set initial position
        mock_bot._last_positions = {"BTC-USD": {"quantity": "0.5", "side": "LONG"}}

        # Position is now closed (empty list)
        mock_bot.broker.list_positions.return_value = []

        monitor = SystemMonitor(mock_bot)

        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should detect position closed
        mock_bot.event_store.append_metric.assert_called()

    @pytest.mark.asyncio
    async def test_position_reconciliation_broker_error(self, mock_bot):
        """Should handle broker errors during reconciliation."""
        mock_bot.broker.list_positions.side_effect = Exception("API error")

        monitor = SystemMonitor(mock_bot)

        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should not crash

    @pytest.mark.asyncio
    @patch("bot_v2.orchestration.system_monitor._get_plog")
    async def test_position_reconciliation_plog_error(self, mock_plog, mock_bot):
        """Should handle plog errors during reconciliation."""
        mock_bot._last_positions = {"BTC-USD": {"quantity": "0.5", "side": "LONG"}}

        new_position = Mock()
        new_position.symbol = "BTC-USD"
        new_position.quantity = Decimal("1.0")
        new_position.side = "LONG"

        mock_bot.broker.list_positions.return_value = [new_position]
        mock_plog.return_value.log_position_change.side_effect = Exception("Plog error")

        monitor = SystemMonitor(mock_bot)

        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should not crash

    @pytest.mark.asyncio
    async def test_position_reconciliation_event_store_error(self, mock_bot):
        """Should handle event store errors during reconciliation."""
        mock_bot._last_positions = {"BTC-USD": {"quantity": "0.5", "side": "LONG"}}

        new_position = Mock()
        new_position.symbol = "BTC-USD"
        new_position.quantity = Decimal("1.0")
        new_position.side = "LONG"

        mock_bot.broker.list_positions.return_value = [new_position]
        mock_bot.event_store.append_metric.side_effect = Exception("Store error")

        monitor = SystemMonitor(mock_bot)

        task = asyncio.create_task(monitor.run_position_reconciliation(interval_seconds=0.1))
        await asyncio.sleep(0.2)
        mock_bot.running = False
        await task

        # Should not crash
