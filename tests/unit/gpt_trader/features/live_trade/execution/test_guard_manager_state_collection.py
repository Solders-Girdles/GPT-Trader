"""Tests for GuardManager.collect_runtime_guard_state."""

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.guards import RuntimeGuardState


def test_collect_runtime_guard_state_basic(guard_manager, mock_broker):
    mock_balance = MagicMock()
    mock_balance.available = Decimal("5000")
    mock_broker.list_balances.return_value = [mock_balance]
    mock_broker.list_positions.return_value = []

    state = guard_manager.collect_runtime_guard_state()

    assert isinstance(state, RuntimeGuardState)
    assert state.equity == Decimal("1000")
    assert state.balances == [mock_balance]
    assert state.positions == []
    assert state.positions_pnl == {}
    assert state.positions_dict == {}


def test_collect_runtime_guard_state_with_positions(guard_manager, mock_broker, mock_position):
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]

    state = guard_manager.collect_runtime_guard_state()

    assert len(state.positions) == 1
    assert "BTC-PERP" in state.positions_pnl
    assert "BTC-PERP" in state.positions_dict


def test_collect_runtime_guard_state_uses_broker_pnl(guard_manager, mock_broker, mock_position):
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]
    mock_broker.get_position_pnl.return_value = {
        "realized_pnl": "500",
        "unrealized_pnl": "200",
    }

    state = guard_manager.collect_runtime_guard_state()

    assert state.positions_pnl["BTC-PERP"]["realized_pnl"] == Decimal("500")
    assert state.positions_pnl["BTC-PERP"]["unrealized_pnl"] == Decimal("200")


def test_collect_runtime_guard_state_fallback_equity(guard_manager, mock_broker):
    mock_balance = MagicMock()
    mock_balance.available = Decimal("5000")
    mock_broker.list_balances.return_value = [mock_balance]
    guard_manager._calculate_equity = MagicMock(return_value=(Decimal("0"), [], Decimal("0")))

    state = guard_manager.collect_runtime_guard_state()

    assert state.equity == Decimal("5000")


def test_collect_runtime_guard_state_handles_position_errors(guard_manager, mock_broker):
    bad_position = MagicMock()
    bad_position.symbol = "BAD-PERP"
    bad_position.entry_price = "invalid"
    bad_position.mark_price = "invalid"
    del bad_position.quantity

    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [bad_position]

    state = guard_manager.collect_runtime_guard_state()

    assert isinstance(state, RuntimeGuardState)
