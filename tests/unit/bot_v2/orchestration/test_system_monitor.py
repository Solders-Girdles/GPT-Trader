"""
Tests for SystemMonitor.

Tests status logging, metrics publishing, configuration monitoring,
and position reconciliation.
"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision
from bot_v2.orchestration.configuration import Profile
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
    state = PerpsBotRuntimeState([])
    bot.runtime_state = state
    bot.last_decisions = state.last_decisions
    bot.order_stats = state.order_stats
    bot.start_time = datetime.now(timezone.utc)
    bot.running = True
    bot._last_positions = state.last_positions
    bot.apply_config_change = Mock()
    return bot


@pytest.fixture
def mock_account_telemetry():
    """Create mock AccountTelemetryService."""
    service = Mock()
    service.latest_snapshot = {
        "total_value": 10000,
        "unrealized_pnl": 500,
    }
    return service


@pytest.fixture
def monitor(mock_bot, mock_account_telemetry):
    """Create SystemMonitor instance."""
    return SystemMonitor(bot=mock_bot, account_telemetry=mock_account_telemetry)


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


class TestSystemMonitorInitialization:
    """Test SystemMonitor initialization."""

    def test_initialization_with_bot(self, mock_bot, mock_account_telemetry):
        """Test monitor initializes with bot reference."""
        monitor = SystemMonitor(bot=mock_bot, account_telemetry=mock_account_telemetry)

        assert monitor._bot == mock_bot
        assert monitor._account_telemetry == mock_account_telemetry

    def test_initialization_without_account_telemetry(self, mock_bot):
        """Test monitor initializes without account telemetry."""
        monitor = SystemMonitor(bot=mock_bot)

        assert monitor._bot == mock_bot
        assert monitor._account_telemetry is None


class TestAttachAccountTelemetry:
    """Test attach_account_telemetry method."""

    def test_attaches_account_telemetry_service(self, monitor):
        """Test attaches account telemetry service."""
        new_service = Mock()

        monitor.attach_account_telemetry(new_service)

        assert monitor._account_telemetry == new_service


class TestLogStatus:
    """Test log_status async method."""

    @pytest.mark.asyncio
    async def test_fetches_positions_and_balances(
        self, monitor, mock_bot, test_balance, test_position
    ):
        """Test fetches positions and balances from broker."""
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        with patch.object(monitor._metrics_publisher, "publish"):
            await monitor.log_status()

        mock_bot.broker.list_positions.assert_called_once()
        mock_bot.broker.list_balances.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculates_equity_from_usd_balance(self, monitor, mock_bot, test_balance):
        """Test calculates equity from USD balance."""
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        metrics_published = None

        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        assert metrics_published is not None
        assert metrics_published["equity"] == 10000

    @pytest.mark.asyncio
    async def test_includes_positions_in_metrics(
        self, monitor, mock_bot, test_balance, test_position
    ):
        """Test includes positions in metrics payload."""
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        metrics_published = None

        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        assert len(metrics_published["positions"]) == 1
        position = metrics_published["positions"][0]
        assert position["symbol"] == "BTC-PERP"
        assert position["quantity"] == 0.5
        assert position["side"] == "long"

    @pytest.mark.asyncio
    async def test_includes_decisions_in_metrics(self, monitor, mock_bot, test_balance):
        """Test includes last decisions in metrics."""
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])
        decisions = mock_bot.runtime_state.last_decisions
        decisions.clear()
        decisions["BTC-PERP"] = Decision(action=Action.BUY, reason="test_signal")

        metrics_published = None

        def capture_metrics(metrics):
            nonlocal metrics_published
            metrics_published = metrics

        with patch.object(monitor._metrics_publisher, "publish", side_effect=capture_metrics):
            await monitor.log_status()

        assert "BTC-PERP" in metrics_published["decisions"]
        assert metrics_published["decisions"]["BTC-PERP"]["action"] == "buy"
        assert metrics_published["decisions"]["BTC-PERP"]["reason"] == "test_signal"

    @pytest.mark.asyncio
    async def test_handles_position_fetch_failure(self, monitor, mock_bot, test_balance):
        """Test handles position fetch failure gracefully."""
        mock_bot.broker.list_positions = Mock(side_effect=Exception("fetch failed"))
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.orders_store.get_open_orders = Mock(return_value=[])

        with patch.object(monitor._metrics_publisher, "publish"):
            await monitor.log_status()

        # Should not raise


class TestMetricsPublisher:
    """Tests for MetricsPublisher component."""

    def test_appends_to_event_store(self, tmp_path, mock_bot):
        publisher = MetricsPublisher(
            event_store=mock_bot.event_store,
            bot_id=mock_bot.bot_id,
            profile=mock_bot.config.profile.value,
            base_dir=tmp_path,
        )
        with patch("bot_v2.orchestration.system_monitor_metrics._get_plog") as mock_plog:
            mock_plog.return_value.log_event = Mock()
            publisher.publish({"equity": 10000})

        mock_bot.event_store.append_metric.assert_called_once()
        call_kwargs = mock_bot.event_store.append_metric.call_args.kwargs
        assert call_kwargs["bot_id"] == mock_bot.bot_id

    def test_writes_metrics_snapshot(self, tmp_path, mock_bot):
        publisher = MetricsPublisher(
            event_store=mock_bot.event_store,
            bot_id=mock_bot.bot_id,
            profile=mock_bot.config.profile.value,
            base_dir=tmp_path,
        )
        with patch("bot_v2.orchestration.system_monitor_metrics._get_plog") as mock_plog:
            mock_plog.return_value.log_event = Mock()
            publisher.publish({"equity": 42})

        metrics_file = tmp_path / "perps_bot" / mock_bot.config.profile.value / "metrics.json"
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        assert data["equity"] == 42

    def test_write_health_status(self, tmp_path, mock_bot):
        publisher = MetricsPublisher(
            event_store=mock_bot.event_store,
            bot_id=mock_bot.bot_id,
            profile=mock_bot.config.profile.value,
            base_dir=tmp_path,
        )
        publisher.write_health_status(ok=True, message="ok", error="")
        status_file = tmp_path / "perps_bot" / mock_bot.config.profile.value / "health.json"
        assert status_file.exists()
        payload = json.loads(status_file.read_text())
        assert payload["ok"] is True
        assert payload["message"] == "ok"


class TestCheckConfigUpdates:
    """Test check_config_updates method."""

    def test_returns_early_when_no_controller(self, monitor, mock_bot):
        """Test returns early when no config controller."""
        mock_bot.config_controller = None

        # Should not raise
        monitor.check_config_updates()

    def test_returns_when_no_changes(self, monitor, mock_bot):
        """Test returns when no config changes detected."""
        controller = Mock()
        controller.refresh_if_changed = Mock(return_value=None)
        mock_bot.config_controller = controller

        monitor.check_config_updates()

        controller.refresh_if_changed.assert_called_once()
        mock_bot.apply_config_change.assert_not_called()

    def test_handles_validation_error(self, monitor, mock_bot):
        """Test handles config validation error."""
        from bot_v2.orchestration.configuration import ConfigValidationError

        controller = Mock()
        controller.refresh_if_changed = Mock(side_effect=ConfigValidationError("invalid config"))
        mock_bot.config_controller = controller

        # Should not raise
        monitor.check_config_updates()

        mock_bot.apply_config_change.assert_not_called()

    def test_applies_config_change_when_detected(self, monitor, mock_bot):
        """Test applies config change when detected."""
        change = Mock()
        change.diff = {"short_ma": {"old": 10, "new": 15}}

        controller = Mock()
        controller.current = Mock()
        controller.current.profile = Profile.PROD
        controller.refresh_if_changed = Mock(return_value=change)
        controller.consume_pending_change = Mock()
        mock_bot.config_controller = controller

        monitor.check_config_updates()

        mock_bot.apply_config_change.assert_called_once_with(change)
        controller.consume_pending_change.assert_called_once()


class TestWriteHealthStatus:
    """Test write_health_status method."""

    def test_writes_ok_status(self, monitor, mock_bot):
        """Test writes OK health status."""
        with patch.object(monitor._metrics_publisher, "write_health_status") as mock_write:
            monitor.write_health_status(ok=True, message="All systems operational")

        mock_write.assert_called_once_with(ok=True, message="All systems operational", error="")

    def test_writes_error_status(self, monitor, mock_bot):
        """Test writes error health status."""
        with patch.object(monitor._metrics_publisher, "write_health_status") as mock_write:
            monitor.write_health_status(ok=False, error="Connection timeout")

        mock_write.assert_called_once_with(ok=False, message="", error="Connection timeout")


class TestPositionReconciliation:
    """Test position reconciliation async loop."""

    @pytest.mark.asyncio
    async def test_initializes_positions_on_first_run(self, monitor, mock_bot, test_position):
        """Test initializes positions on first run."""
        monitor._position_reconciler._fetch_positions = AsyncMock(return_value=[test_position])

        # Run one iteration
        iteration_count = [0]
        original_sleep = asyncio.sleep

        async def limited_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 1:
                mock_bot.running = False
            await original_sleep(0)

        with patch(
            "bot_v2.orchestration.system_monitor_positions.asyncio.sleep", side_effect=limited_sleep
        ):
            await monitor.run_position_reconciliation(interval_seconds=1)

        assert "BTC-PERP" in mock_bot.runtime_state.last_positions

    @pytest.mark.asyncio
    async def test_detects_new_position(self, monitor, mock_bot, test_position):
        """Test detects new position."""
        state = mock_bot.runtime_state
        state.last_positions.clear()
        state.last_positions["ETH-PERP"] = {"quantity": "1.0", "side": "long"}
        monitor._position_reconciler._fetch_positions = AsyncMock(return_value=[test_position])

        # Run one iteration
        iteration_count = [0]
        original_sleep = asyncio.sleep

        async def limited_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 1:
                mock_bot.running = False
            await original_sleep(0)

        with patch(
            "bot_v2.orchestration.system_monitor_positions.asyncio.sleep", side_effect=limited_sleep
        ):
            await monitor.run_position_reconciliation(interval_seconds=1)

        # Position should be logged
        mock_bot.event_store.append_metric.assert_called()
        call_kwargs = mock_bot.event_store.append_metric.call_args.kwargs
        assert call_kwargs["metrics"]["event_type"] == "position_drift"

    @pytest.mark.asyncio
    async def test_detects_position_quantity_change(self, monitor, mock_bot, test_position):
        """Test detects quantity change in position."""
        state = mock_bot.runtime_state
        state.last_positions.clear()
        state.last_positions["BTC-PERP"] = {"quantity": "0.3", "side": "long"}
        test_position.quantity = Decimal("0.5")
        monitor._position_reconciler._fetch_positions = AsyncMock(return_value=[test_position])

        # Run one iteration
        iteration_count = [0]
        original_sleep = asyncio.sleep

        async def limited_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 1:
                mock_bot.running = False
            await original_sleep(0)

        with patch(
            "bot_v2.orchestration.system_monitor_positions.asyncio.sleep", side_effect=limited_sleep
        ):
            await monitor.run_position_reconciliation(interval_seconds=1)

        # Change should be logged
        mock_bot.event_store.append_metric.assert_called()

    @pytest.mark.asyncio
    async def test_detects_closed_position(self, monitor, mock_bot):
        """Test detects when position is closed."""
        state = mock_bot.runtime_state
        state.last_positions.clear()
        state.last_positions["BTC-PERP"] = {"quantity": "0.5", "side": "long"}
        monitor._position_reconciler._fetch_positions = AsyncMock(return_value=[])

        # Run one iteration
        iteration_count = [0]
        original_sleep = asyncio.sleep

        async def limited_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 1:
                mock_bot.running = False
            await original_sleep(0)

        with patch(
            "bot_v2.orchestration.system_monitor_positions.asyncio.sleep", side_effect=limited_sleep
        ):
            await monitor.run_position_reconciliation(interval_seconds=1)

        # Closure should be logged
        mock_bot.event_store.append_metric.assert_called()
        call_kwargs = mock_bot.event_store.append_metric.call_args.kwargs
        changes = call_kwargs["metrics"]["changes"]
        assert "BTC-PERP" in changes
        assert changes["BTC-PERP"]["new"] == {}

    @pytest.mark.asyncio
    async def test_handles_position_fetch_failure(self, monitor, mock_bot):
        """Test handles position fetch failure gracefully."""
        monitor._position_reconciler._fetch_positions = AsyncMock(
            side_effect=Exception("fetch failed")
        )

        # Run one iteration
        iteration_count = [0]
        original_sleep = asyncio.sleep

        async def limited_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 1:
                mock_bot.running = False
            await original_sleep(0)

        with patch(
            "bot_v2.orchestration.system_monitor_positions.asyncio.sleep", side_effect=limited_sleep
        ):
            # Should not raise
            await monitor.run_position_reconciliation(interval_seconds=1)
