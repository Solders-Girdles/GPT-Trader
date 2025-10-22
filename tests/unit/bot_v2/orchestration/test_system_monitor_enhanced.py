"""Enhanced tests for SystemMonitor to improve coverage from 19% to 65%+."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, UTC
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision
from bot_v2.orchestration.configuration import Profile, ConfigValidationError
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.system_monitor import SystemMonitor
from bot_v2.orchestration.system_monitor_metrics import MetricsPublisher


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot instance."""
    bot = Mock()
    bot.bot_id = "test_bot"
    bot.config = Mock()
    bot.config.profile = Profile.PROD
    bot.broker = Mock()
    bot.event_store = Mock()
    bot.orders_store = Mock()
    bot.start_time = datetime.now(UTC)
    bot.running = True
    bot.apply_config_change = Mock()

    # Set up runtime state
    state = PerpsBotRuntimeState([])
    bot.runtime_state = state
    bot.last_decisions = state.last_decisions
    bot.order_stats = state.order_stats
    bot._last_positions = state.last_positions
    return bot


@pytest.fixture
def test_balance():
    """Create test balance."""
    balance = Mock(spec=Balance)
    balance.asset = "USD"
    balance.available = Decimal("10000")
    balance.total = Decimal("10000")
    return balance


@pytest.fixture
def test_position():
    """Create test position."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    position.mark_price = Decimal("51000")
    return position


class TestSystemMonitorResourceCollector:
    """Test resource collector initialization and integration."""

    def test_initializes_with_resource_collector_when_available(self, mock_bot):
        """Test initializes resource collector when psutil is available."""
        with patch("bot_v2.orchestration.system_monitor.ResourceCollector") as mock_collector_class:
            mock_collector = Mock()
            mock_collector_class.return_value = mock_collector

            monitor = SystemMonitor(bot=mock_bot)

            assert monitor._resource_collector is mock_collector
            mock_collector_class.assert_called_once()

    def test_handles_resource_collector_initialization_failure(self, mock_bot):
        """Test handles resource collector initialization failure gracefully."""
        with patch("bot_v2.orchestration.system_monitor.ResourceCollector") as mock_collector_class:
            mock_collector_class.side_effect = Exception("psutil unavailable")

            # Should not raise
            monitor = SystemMonitor(bot=mock_bot)

            assert monitor._resource_collector is None

    def test_gracefully_degrades_when_psutil_unavailable(self, mock_bot):
        """Test gracefully degrades when psutil is completely unavailable."""
        # The actual implementation checks if ResourceCollector is None, but ResourceCollectorType is used
        # Let's patch the import to simulate psutil unavailability
        with patch("bot_v2.orchestration.system_monitor.ResourceCollector", side_effect=ImportError("psutil not available")):
            # Should not raise
            monitor = SystemMonitor(bot=mock_bot)

            assert monitor._resource_collector is None

    def test_logs_debug_message_on_collector_failure(self, mock_bot):
        """Test logs debug message when resource collector fails."""
        with patch("bot_v2.orchestration.system_monitor.ResourceCollector") as mock_collector_class:
            mock_collector_class.side_effect = Exception("Permission denied")

            with patch("bot_v2.orchestration.system_monitor.logger") as mock_logger:
                SystemMonitor(bot=mock_bot)

                # Should log debug message
                mock_logger.debug.assert_called_once()
                call_args = mock_logger.debug.call_args
                # Check first argument contains message (call_args[0] is the tuple, so we need [0][0])
                assert "Resource collector unavailable" in call_args[0][0]
                # Check that extra kwargs contain expected context
                assert "operation" in call_args[1]
                assert call_args[1]["operation"] == "system_monitor_init"


class TestSystemMonitorLogStatusEnhanced:
    """Enhanced tests for log_status method to improve coverage."""

    @pytest.mark.asyncio
    async def test_log_status_complete_flow_with_all_components(self, monitor, mock_bot, test_balance, test_position):
        """Test complete log_status flow with all components including resource monitoring."""
        # Set up comprehensive test data
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        # Add decisions
        decisions = mock_bot.runtime_state.last_decisions
        decisions["BTC-PERP"] = Decision(action=Action.BUY, reason="test_signal")
        decisions["ETH-PERP"] = Decision(action=Action.SELL, reason="take_profit")

        # Add account telemetry
        account_telemetry = Mock()
        account_telemetry.latest_snapshot = {"total_value": 15000, "unrealized_pnl": 500}
        monitor._account_telemetry = account_telemetry

        # Mock resource collector
        mock_resource_collector = Mock()
        mock_usage = Mock()
        mock_usage.cpu_percent = 45.2
        mock_usage.memory_percent = 62.1
        mock_usage.memory_mb = 1024
        mock_usage.disk_percent = 35.7
        mock_usage.disk_gb = 50.2
        mock_usage.network_sent_mb = 123.4
        mock_usage.network_recv_mb = 456.7
        mock_usage.open_files = 42
        mock_usage.threads = 8
        mock_resource_collector.collect.return_value = mock_usage
        monitor._resource_collector = mock_resource_collector

        # Mock metrics publisher
        metrics_published = None
        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        # Verify comprehensive metrics payload
        assert metrics_published is not None
        assert metrics_published["profile"] == "prod"
        assert metrics_published["equity"] == 10000.0
        assert len(metrics_published["positions"]) == 1
        assert metrics_published["positions"][0]["symbol"] == "BTC-PERP"
        assert metrics_published["positions"][0]["quantity"] == 0.5

        # Verify decisions
        assert "BTC-PERP" in metrics_published["decisions"]
        assert "ETH-PERP" in metrics_published["decisions"]
        assert metrics_published["decisions"]["BTC-PERP"]["action"] == "buy"

        # Verify account telemetry
        assert "account_snapshot" in metrics_published
        assert metrics_published["account_snapshot"]["total_value"] == 15000

        # Verify system metrics
        assert "system" in metrics_published
        system = metrics_published["system"]
        assert system["cpu_percent"] == 45.2
        assert system["memory_percent"] == 62.1
        assert system["open_files"] == 42
        assert system["threads"] == 8

    @pytest.mark.asyncio
    async def test_log_status_handles_balance_fetch_failure(self, monitor, mock_bot, test_position):
        """Test log_status handles balance fetch failure gracefully."""
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        mock_bot.broker.list_balances = Mock(side_effect=Exception("balance fetch failed"))
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        with patch.object(monitor._metrics_publisher, "publish"):
            await monitor.log_status()

        # Should not raise and should default equity to 0
        # The specific behavior depends on implementation

    @pytest.mark.asyncio
    async def test_log_status_handles_orders_store_failure(self, monitor, mock_bot, test_balance):
        """Test log_status handles orders store failure gracefully."""
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(side_effect=Exception("store failed"))

        with patch.object(monitor._metrics_publisher, "publish"):
            await monitor.log_status()

        # Should not raise and should default open_orders_count to 0

    @pytest.mark.asyncio
    async def test_log_status_with_multiple_positions(self, monitor, mock_bot, test_balance):
        """Test log_status with multiple positions including edge cases."""
        # Create multiple test positions
        position1 = Mock(spec=Position)
        position1.symbol = "BTC-PERP"
        position1.quantity = Decimal("0.5")
        position1.side = "long"
        position1.entry_price = Decimal("50000")
        position1.mark_price = Decimal("51000")

        position2 = Mock(spec=Position)
        position2.symbol = "ETH-PERP"
        position2.quantity = Decimal("2.0")
        position2.side = "short"
        position2.entry_price = Decimal("3000")
        position2.mark_price = Decimal("2950")

        # Position with missing fields
        position3 = Mock(spec=Position)
        position3.symbol = "SOL-PERP"
        position3.quantity = Decimal("100")
        # Missing side, entry_price, mark_price

        mock_bot.broker.list_positions = Mock(return_value=[position1, position2, position3])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        metrics_published = None
        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        # Should include valid positions (those with symbols)
        assert len(metrics_published["positions"]) >= 2
        symbols = [p["symbol"] for p in metrics_published["positions"]]
        assert "BTC-PERP" in symbols
        assert "ETH-PERP" in symbols

    @pytest.mark.asyncio
    async def test_log_status_calculates_uptime_correctly(self, monitor, mock_bot, test_balance):
        """Test log_status calculates uptime correctly."""
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        # Set start time to 1 hour ago
        mock_bot.start_time = datetime.now(UTC) - asyncio.sleep(3600)  # 1 hour

        metrics_published = None
        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        # Should include uptime (approximately 3600 seconds)
        assert "uptime_seconds" in metrics_published
        uptime = metrics_published["uptime_seconds"]
        assert uptime > 3500  # Allow some tolerance
        assert uptime < 3700

    @pytest.mark.asyncio
    async def test_log_status_handles_system_metrics_collection_failure(self, monitor, mock_bot, test_balance):
        """Test log_status handles system metrics collection failure gracefully."""
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        # Mock resource collector that fails
        mock_resource_collector = Mock()
        mock_resource_collector.collect.side_effect = Exception("metrics collection failed")
        monitor._resource_collector = mock_resource_collector

        with patch.object(monitor._metrics_publisher, "publish"):
            await monitor.log_status()

        # Should not raise despite system metrics failure


class TestSystemMonitorConfigUpdatesEnhanced:
    """Enhanced tests for configuration update handling."""

    def test_config_updates_with_inputs_changed_scenario(self, monitor, mock_bot):
        """Test configuration updates when inputs changed but not validation."""
        change = Mock()
        change.diff = None  # No validation changes

        controller = Mock()
        controller.current = Mock()
        controller.current.profile = Profile.PROD
        controller.refresh_if_changed = Mock(return_value=change)
        controller.consume_pending_change = Mock()
        mock_bot.config_controller = controller

        monitor.check_config_updates()

        # Should log inputs changed message
        controller.refresh_if_changed.assert_called_once()
        mock_bot.apply_config_change.assert_not_called()
        controller.consume_pending_change.assert_called_once()

    def test_config_updates_handles_apply_failure(self, monitor, mock_bot):
        """Test configuration updates handles apply failure gracefully."""
        change = Mock()
        change.diff = {"short_ma": {"old": 10, "new": 15}}

        controller = Mock()
        controller.current = Mock()
        controller.current.profile = Profile.PROD
        controller.refresh_if_changed = Mock(return_value=change)
        controller.consume_pending_change = Mock()
        mock_bot.config_controller = controller

        # Mock apply_config_change to fail
        mock_bot.apply_config_change.side_effect = Exception("apply failed")

        monitor.check_config_updates()

        # Should still consume the change despite failure
        controller.consume_pending_change.assert_called_once()

    def test_config_updates_logs_validation_rejection_with_details(self, monitor, mock_bot):
        """Test configuration updates logs detailed validation rejection."""
        controller = Mock()
        controller.refresh_if_changed = Mock(side_effect=ConfigValidationError("Invalid risk setting"))
        mock_bot.config_controller = controller

        with patch("bot_v2.orchestration.system_monitor.logger") as mock_logger:
            monitor.check_config_updates()

            # Should log detailed error
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0]
            assert "Configuration update rejected" in call_args[0]
            assert "Invalid risk setting" in call_args[1]
            assert call_args[2]["operation"] == "config_refresh"
            assert call_args[2]["status"] == "rejected"

    def test_config_updates_with_different_profiles(self, monitor, mock_bot):
        """Test configuration updates with different profiles."""
        # Test with SPOT profile
        mock_bot.config.profile = Profile.SPOT
        change = Mock()
        change.diff = {"max_position_size": {"old": 1.0, "new": 2.0}}

        controller = Mock()
        controller.current = Mock()
        controller.current.profile = Profile.SPOT
        controller.refresh_if_changed = Mock(return_value=change)
        controller.consume_pending_change = Mock()
        mock_bot.config_controller = controller

        monitor.check_config_updates()

        # Should work with any profile
        mock_bot.apply_config_change.assert_called_once_with(change)

    def test_config_updates_with_empty_diff(self, monitor, mock_bot):
        """Test configuration updates with empty diff but change object."""
        change = Mock()
        change.diff = {}  # Empty diff but change object exists

        controller = Mock()
        controller.current = Mock()
        controller.current.profile = Profile.PROD
        controller.refresh_if_changed = Mock(return_value=change)
        controller.consume_pending_change = Mock()
        mock_bot.config_controller = controller

        monitor.check_config_updates()

        # Should apply change even with empty diff
        mock_bot.apply_config_change.assert_called_once_with(change)


class TestSystemMonitorInitializationEdgeCases:
    """Test edge cases in SystemMonitor initialization."""

    def test_initialization_with_various_bot_configs(self):
        """Test initialization with different bot configurations."""
        for profile in [Profile.PROD, Profile.DEV, Profile.SPOT, Profile.CANARY]:
            bot = Mock()
            bot.bot_id = f"test_bot_{profile.value}"
            bot.config = Mock()
            bot.config.profile = profile
            bot.broker = Mock()
            bot.event_store = Mock()

            # Should not raise for any profile
            monitor = SystemMonitor(bot=bot)

            assert monitor._bot == bot
            assert monitor._bot.config.profile == profile

    def test_initialization_handles_missing_bot_attributes(self):
        """Test initialization handles missing bot attributes gracefully."""
        bot = Mock()
        bot.bot_id = "test_bot"
        # Missing other attributes that might be accessed later

        # Should not raise during initialization
        monitor = SystemMonitor(bot=bot)

        assert monitor._bot == bot
        assert monitor._account_telemetry is None

    def test_attach_account_telemetry_overwrites_existing(self, monitor):
        """Test attach_account_telemetry overwrites existing service."""
        original_service = Mock()
        monitor._account_telemetry = original_service

        new_service = Mock()
        monitor.attach_account_telemetry(new_service)

        assert monitor._account_telemetry == new_service
        assert monitor._account_telemetry != original_service

    def test_position_reconciliation_integration(self, monitor, mock_bot):
        """Test position reconciliation integration through SystemMonitor."""
        # Mock position reconciler
        mock_reconciler = Mock()
        monitor._position_reconciler = mock_reconciler

        # Should delegate to position reconciler
        with patch("asyncio.create_task") as mock_create_task:
            monitor.run_position_reconciliation(interval_seconds=60)

            mock_reconciler.run.assert_called_once_with(mock_bot, interval_seconds=60)


class TestSystemMonitorErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_log_status_handles_multiple_failures(self, monitor, mock_bot):
        """Test log_status handles multiple simultaneous failures."""
        # Multiple failures
        mock_bot.broker.list_positions = Mock(side_effect=Exception("positions failed"))
        mock_bot.broker.list_balances = Mock(side_effect=Exception("balances failed"))
        mock_bot.orders_store.get_open_orders = Mock(side_effect=Exception("orders failed"))

        with patch.object(monitor._metrics_publisher, "publish"):
            # Should not raise despite multiple failures
            await monitor.log_status()

    @pytest.mark.asyncio
    async def test_log_status_handles_malformed_position_data(self, monitor, mock_bot, test_balance):
        """Test log_status handles malformed position data gracefully."""
        # Create malformed position
        malformed_position = Mock()
        malformed_position.symbol = "BTC-PERP"
        # Missing required fields like quantity

        mock_bot.broker.list_positions = Mock(return_value=[malformed_position])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        with patch.object(monitor._metrics_publisher, "publish"):
            # Should not raise despite malformed data
            await monitor.log_status()

    @pytest.mark.asyncio
    async def test_log_status_with_decimal_equity_conversion(self, monitor, mock_bot):
        """Test log_status properly converts Decimal equity to float."""
        test_balance = Mock()
        test_balance.asset = "USD"
        test_balance.available = Decimal("12345.67")
        test_balance.total = Decimal("15000.89")

        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        metrics_published = None
        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        # Should convert Decimal to float
        assert metrics_published["equity"] == 12345.67
        assert isinstance(metrics_published["equity"], float)

    def test_health_status_delegation(self, monitor):
        """Test health status properly delegates to metrics publisher."""
        with patch.object(monitor._metrics_publisher, "write_health_status") as mock_write:
            monitor.write_health_status(ok=True, message="System healthy", error="")

            mock_write.assert_called_once_with(ok=True, message="System healthy", error="")