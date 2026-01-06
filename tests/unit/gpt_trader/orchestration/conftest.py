"""
Reusable fixtures for orchestration coordinator tests.

Provides mocks for external dependencies, event bus, guard manager,
trade service, and runtime state to enable isolated unit testing.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.config.types import Profile
from gpt_trader.core import Balance, Position
from gpt_trader.features.live_trade.engines.base import CoordinatorContext

# Mock imports for missing types
try:
    from gpt_trader.features.risk.circuit_breaker import CircuitBreakerOutcome
except ImportError:
    CircuitBreakerOutcome = Mock()

try:
    from gpt_trader.orchestration.system_monitor_metrics import SystemMonitor
except ImportError:
    SystemMonitor = Mock()


@pytest.fixture
def test_bot_config(tmp_path) -> BotConfig:
    """Provide a BotConfig for orchestration tests."""
    return BotConfig(
        symbols=["BTC-PERP"],
        profile=Profile.DEV,
        mock_broker=True,
        dry_run=True,
        runtime_root=str(tmp_path),
    )


@pytest.fixture
def fake_guard_manager():
    return Mock()


@pytest.fixture
def fake_event_bus():
    return Mock()


@pytest.fixture
def fake_trade_service():
    return Mock()


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

    # Create mock container
    container = Mock()
    container.config = config

    broker = Mock()
    risk_manager = Mock()
    risk_manager.config = Mock()
    risk_manager.config.kill_switch_enabled = False
    risk_manager.check_volatility_circuit_breaker = Mock(
        return_value=CircuitBreakerOutcome(triggered=False, action=None, reason=None)
    )
    risk_manager.check_mark_staleness = Mock(return_value=False)

    event_store = Mock()

    context = CoordinatorContext(
        config=config,
        container=container,
        event_store=event_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="test-bot",
        runtime_state=fake_runtime_state,
    )
    return context


@pytest.fixture
def monitor(mock_bot):
    """Create a SystemMonitor instance for testing."""
    return SystemMonitor(bot=mock_bot, account_telemetry=None)


@pytest.fixture
def mock_bot():
    return Mock()
