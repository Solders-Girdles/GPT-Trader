"""Shared fixtures for system monitor tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.system_monitor import SystemMonitor
from bot_v2.orchestration.system_monitor_metrics import MetricsPublisher
from bot_v2.orchestration.system_monitor_positions import PositionReconciler


@pytest.fixture
def mock_bot():
    """Create realistic mock PerpsBot instance for testing."""
    bot = MagicMock()
    bot.bot_id = "test_bot"
    bot.config = MagicMock()
    bot.config.profile = Profile.PROD

    # Broker setup
    bot.broker = MagicMock()
    bot.event_store = MagicMock()
    bot.orders_store = MagicMock()

    # Runtime state
    state = PerpsBotRuntimeState([])
    bot.runtime_state = state
    bot.last_decisions = state.last_decisions
    bot.order_stats = state.order_stats
    bot.start_time = datetime.now(UTC)
    bot.running = True

    # Config controller
    bot.config_controller = MagicMock()
    bot.apply_config_change = MagicMock()

    return bot


@pytest.fixture
def fake_account_telemetry():
    """Create mock AccountTelemetryService."""
    service = MagicMock()
    service.latest_snapshot = {
        "total_value": Decimal("10000"),
        "available_balance": Decimal("8500"),
        "margin_used": Decimal("1500"),
    }
    return service


@pytest.fixture
def fake_metrics_publisher():
    """Mock MetricsPublisher with call tracking."""
    publisher = MagicMock(spec=MetricsPublisher)
    publisher.publish = MagicMock()
    publisher.write_health_status = MagicMock()
    return publisher


@pytest.fixture
def fake_position_reconciler():
    """Mock PositionReconciler with call tracking."""
    reconciler = MagicMock(spec=PositionReconciler)
    reconciler.run = AsyncMock()
    return reconciler


@pytest.fixture(autouse=True)
def patch_asyncio_to_thread():
    """Patch asyncio.to_thread for deterministic async testing."""
    def sync_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("bot_v2.orchestration.system_monitor.asyncio.to_thread", side_effect=sync_to_thread):
        yield


@pytest.fixture
def sample_positions():
    """Create sample position data for testing."""
    return [
        MagicMock(
            symbol="BTC-PERP",
            quantity=Decimal("0.5"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50100"),
        ),
        MagicMock(
            symbol="ETH-PERP",
            quantity=Decimal("2.0"),
            side="short",
            entry_price=Decimal("3000"),
            mark_price=Decimal("2995"),
        ),
    ]


@pytest.fixture
def sample_balances():
    """Create sample balance data for testing."""
    return [
        MagicMock(asset="USD", available=Decimal("8500"), total=Decimal("10000")),
        MagicMock(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1")),
    ]


@pytest.fixture
def sample_decisions():
    """Create sample decision data for testing."""
    from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

    return {
        "BTC-PERP": Decision(action=Action.HOLD, reason="within_threshold"),
        "ETH-PERP": Decision(action=Action.BUY, reason="below_entry"),
    }


@pytest.fixture
def system_monitor(mock_bot, fake_account_telemetry, fake_metrics_publisher, fake_position_reconciler):
    """Real SystemMonitor instance with external dependencies patched."""
    with patch("bot_v2.orchestration.system_monitor.MetricsPublisher", return_value=fake_metrics_publisher), \
         patch("bot_v2.orchestration.system_monitor.PositionReconciler", return_value=fake_position_reconciler):
        return SystemMonitor(bot=mock_bot, account_telemetry=fake_account_telemetry)


@pytest.fixture
def system_monitor_no_resource_collector(mock_bot, fake_account_telemetry):
    """SystemMonitor instance for testing psutil unavailable scenarios."""
    with patch("bot_v2.orchestration.system_monitor.ResourceCollector", side_effect=ImportError("No psutil")):
        return SystemMonitor(bot=mock_bot, account_telemetry=fake_account_telemetry)


@pytest.fixture
def fake_resource_collector():
    """Mock ResourceCollector with realistic data."""
    collector = MagicMock()

    # Simulate system resource usage
    usage = SimpleNamespace()
    usage.cpu_percent = 15.5
    usage.memory_percent = 45.2
    usage.memory_mb = 2048.0
    usage.disk_percent = 67.8
    usage.disk_gb = 125.5
    usage.network_sent_mb = 1024.5
    usage.network_recv_mb = 2048.2
    usage.open_files = 156
    usage.threads = 24

    collector.collect.return_value = usage
    return collector