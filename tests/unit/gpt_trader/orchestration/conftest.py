"""
Reusable fixtures for orchestration coordinator tests.

Provides mocks for external dependencies, event bus, guard manager,
trade service, and runtime state to enable isolated unit testing.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.brokerages.core.interfaces import Balance, Position
from gpt_trader.orchestration.configuration import BotConfig, Profile
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
@pytest.fixture
def fake_runtime_state():
    """Mock runtime state with common test data."""
    state = Mock()
    state.mark_windows = {"BTC-PERP": [Decimal("50000")] * 35}
    state.mark_lock = Mock()
    state.mark_lock.__enter__ = Mock(return_value=None)
    state.mark_lock.__exit__ = Mock(return_value=None)
    state.last_decisions = {}
    state.order_lock = None
    state.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
    state.exec_engine = None
    state.strategy = Mock()
    state.symbol_strategies = {}
    return state


@pytest.fixture
def test_balance():
    """Standard test balance fixture."""
    balance = Mock(spec=Balance)
    balance.asset = "USDC"
    balance.total = Decimal("10000")
    return balance


@pytest.fixture
def test_position():
    """Standard test position fixture."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    return position


@pytest.fixture
def base_coordinator_context(
    fake_guard_manager, fake_event_bus, fake_trade_service, fake_runtime_state
):
    """Base coordinator context with mocked dependencies."""
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    registry = ServiceRegistry(config=config)

    # Add mocks to registry extras
    registry = registry.with_updates(
        extras={
            "guard_manager": fake_guard_manager,
            "event_bus": fake_event_bus,
            "trade_service": fake_trade_service,
        }
    )

    broker = Mock()
    risk_manager = Mock()
    risk_manager.config = Mock()
    risk_manager.config.kill_switch_enabled = False
    risk_manager.check_volatility_circuit_breaker = Mock(
        return_value=CircuitBreakerOutcome(triggered=False, action=None, reason=None)
    )
    risk_manager.check_mark_staleness = Mock(return_value=False)

    orders_store = Mock()
    event_store = Mock()

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="test-bot",
        runtime_state=fake_runtime_state,
        config_controller=Mock(),
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def monitor(mock_bot):
    """Create a SystemMonitor instance for testing."""
    return SystemMonitor(bot=mock_bot, account_telemetry=None)
