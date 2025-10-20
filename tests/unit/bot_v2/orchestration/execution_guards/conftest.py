"""
Fixtures for execution guards tests.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    IBrokerage,
    Position,
)
from bot_v2.orchestration.execution.guards import GuardManager, RuntimeGuardState


@pytest.fixture
def fake_balance():
    """Create test balance."""
    balance = Mock(spec=Balance)
    balance.asset = "USDC"
    balance.total = Decimal("10000")
    balance.available = Decimal("9500")
    return balance


@pytest.fixture
def fake_position():
    """Create test position."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    position.mark_price = Decimal("51000")
    return position


@pytest.fixture
def fake_broker():
    """Create fake broker with async methods."""
    broker = Mock(spec=IBrokerage)
    broker.list_balances = Mock()
    broker.list_positions = Mock()
    broker.get_candles = Mock()
    broker.get_position_pnl = Mock()
    broker.get_position_risk = Mock()
    broker._mark_cache = Mock()
    broker._mark_cache.get_mark = Mock()
    return broker


@pytest.fixture
def fake_risk_manager():
    """Create fake risk manager."""
    rm = Mock()
    rm.config = Mock()
    rm.config.volatility_window_periods = 20
    rm.last_mark_update = {"BTC-PERP": 1234567890.0}
    rm.track_daily_pnl = Mock(return_value=False)
    rm.check_liquidation_buffer = Mock()
    rm.check_mark_staleness = Mock()
    rm.append_risk_metrics = Mock()
    rm.check_correlation_risk = Mock()
    rm.check_volatility_circuit_breaker = Mock(
        return_value=Mock(triggered=False, to_payload=lambda: {})
    )
    return rm


@pytest.fixture
def equity_calculator():
    """Create equity calculator function."""

    def calc(balances):
        total = sum((b.available for b in balances), Decimal("0"))
        return total, balances, total

    return calc


@pytest.fixture
def cancel_orders_callback():
    """Create cancel orders callback."""
    return Mock(return_value=5)  # Returns number of cancelled orders


@pytest.fixture
def invalidate_cache_callback():
    """Create invalidate cache callback."""
    return Mock()


@pytest.fixture
def guard_manager(
    fake_broker,
    fake_risk_manager,
    equity_calculator,
    cancel_orders_callback,
    invalidate_cache_callback,
):
    """Create GuardManager instance."""
    return GuardManager(
        broker=fake_broker,
        risk_manager=fake_risk_manager,
        equity_calculator=equity_calculator,
        cancel_orders_callback=cancel_orders_callback,
        invalidate_cache_callback=invalidate_cache_callback,
    )


@pytest.fixture
def runtime_guard_state(fake_balance, fake_position):
    """Create RuntimeGuardState instance."""
    return RuntimeGuardState(
        timestamp=1234567890.0,
        balances=[fake_balance],
        equity=Decimal("9500"),
        positions=[fake_position],
        positions_pnl={
            "BTC-PERP": {"realized_pnl": Decimal("100"), "unrealized_pnl": Decimal("500")}
        },
        positions_dict={
            "BTC-PERP": {
                "quantity": Decimal("0.5"),
                "mark": Decimal("51000"),
                "entry": Decimal("50000"),
            }
        },
    )
