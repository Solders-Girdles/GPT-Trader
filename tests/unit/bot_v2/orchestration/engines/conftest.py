"""Shared fixtures for telemetry coordinator tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_v2.config.types import Profile
from bot_v2.orchestration.engines.base import CoordinatorContext
from bot_v2.orchestration.engines.telemetry_coordinator import TelemetryEngine
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def fake_context():
    """Create realistic fake CoordinatorContext for testing."""
    config = MagicMock()
    config.profile = Profile.DEV
    config.perps_enable_streaming = False
    config.perps_stream_level = 1
    config.account_telemetry_interval = 300
    config.short_ma = 10
    config.long_ma = 20
    config.symbols = ["BTC-PERP", "ETH-PERP"]

    registry = MagicMock(spec=ServiceRegistry)
    registry.extras = {}
    registry.risk_manager = MagicMock()
    registry.brokerage = MagicMock()

    broker = MagicMock()
    event_store = MagicMock()
    runtime_state = MagicMock()
    runtime_state.mark_windows = {}
    runtime_state.mark_lock = MagicMock()
    runtime_state.last_decisions = {}
    runtime_state.order_stats = {"attempted": 0, "successful": 0, "failed": 0}

    context = CoordinatorContext(
        bot_id="test_bot",
        config=config,
        registry=registry,
        broker=broker,
        event_store=event_store,
        runtime_state=runtime_state,
        risk_manager=registry.risk_manager,
        symbols=config.symbols,
    )

    return context


@pytest.fixture
def telemetry_coordinator(fake_context):
    """Create TelemetryEngine instance with basic context."""
    coordinator = TelemetryEngine(context=fake_context)
    return coordinator


@pytest.fixture
def mock_broker():
    """Create mock broker for streaming tests."""
    broker = MagicMock()

    # Mock stream methods
    broker.stream_orderbook.return_value = iter(
        [
            {"product_id": "BTC-PERP", "best_bid": "50000", "best_ask": "50100"},
            {"product_id": "ETH-PERP", "best_bid": "3000", "best_ask": "3010"},
        ]
    )

    broker.stream_trades.return_value = iter(
        [
            {"symbol": "BTC-PERP", "price": "50050"},
            {"symbol": "ETH-PERP", "price": "3005"},
        ]
    )

    return broker


@pytest.fixture
def sample_stream_messages():
    """Create sample WebSocket stream messages for testing."""
    return [
        {"product_id": "BTC-PERP", "best_bid": "50000", "best_ask": "50100"},
        {"symbol": "ETH-PERP", "last": "3005"},
        {"product_id": "BTC-PERP", "price": "50025"},
        {"symbol": "ETH-PERP", "bid": "2995", "ask": "3005"},
        # Invalid messages that should be filtered
        None,
        {},
        {"invalid": "message"},
        {"product_id": "", "best_bid": "50000", "best_ask": "50100"},
        {"symbol": "BTC-PERP", "price": "invalid"},
        {"symbol": "BTC-PERP", "price": "-100"},
        {"symbol": "BTC-PERP", "best_bid": "50000"},  # Missing ask
    ]


@pytest.fixture
def mock_account_telemetry():
    """Create mock AccountTelemetryService."""
    service = MagicMock()
    service.supports_snapshots.return_value = True
    service.run = AsyncMock()
    return service


@pytest.fixture
def mock_market_monitor():
    """Create mock MarketActivityMonitor."""
    monitor = MagicMock()
    monitor.record_update = MagicMock()
    return monitor


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = MagicMock()
    manager.record_mark_update = MagicMock(return_value=datetime.now(UTC))
    manager.last_mark_update = {}
    return manager


@pytest.fixture
def telemetry_coordinator_with_services(
    fake_context, mock_account_telemetry, mock_market_monitor, mock_risk_manager
):
    """Create TelemetryEngine with mock services already in registry."""
    # Add mock services to registry extras
    fake_context.registry.extras.update(
        {
            "account_telemetry": mock_account_telemetry,
            "market_monitor": mock_market_monitor,
        }
    )
    fake_context.risk_manager = mock_risk_manager

    coordinator = TelemetryEngine(context=fake_context)
    return coordinator
