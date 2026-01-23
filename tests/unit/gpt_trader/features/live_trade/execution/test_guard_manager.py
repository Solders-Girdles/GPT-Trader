import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.execution.guards import RuntimeGuardState


def test_invalidate_cache_clears_state(guard_manager):
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_dirty = False

    guard_manager.invalidate_cache()

    assert guard_manager._runtime_guard_state is None
    assert guard_manager._runtime_guard_dirty is True
    guard_manager._invalidate_cache_callback.assert_called_once()


def test_invalidate_cache_handles_no_callback(
    mock_broker, mock_risk_manager, mock_equity_calculator
):
    gm = GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=mock_equity_calculator,
        open_orders=[],
        invalidate_cache_callback=None,
    )
    gm._runtime_guard_state = MagicMock()

    gm.invalidate_cache()
    assert gm._runtime_guard_state is None


def test_should_run_full_guard_when_dirty(guard_manager):
    guard_manager._runtime_guard_dirty = True
    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_when_no_state(guard_manager):
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = None
    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_when_interval_elapsed(guard_manager):
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_last_full_ts = time.time() - 120
    guard_manager._runtime_guard_full_interval = 60

    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_returns_false_within_interval(guard_manager):
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_last_full_ts = time.time() - 30
    guard_manager._runtime_guard_full_interval = 60

    assert guard_manager.should_run_full_guard(time.time()) is False


def test_collect_runtime_guard_state_basic(guard_manager, mock_broker):
    mock_balance = MagicMock()
    mock_balance.available = Decimal("5000")
    mock_broker.list_balances.return_value = [mock_balance]
    mock_broker.list_positions.return_value = []

    state = guard_manager.collect_runtime_guard_state()

    assert isinstance(state, RuntimeGuardState)
    assert state.equity == Decimal("1000") and state.balances == [mock_balance]
    assert state.positions == [] and state.positions_pnl == {} and state.positions_dict == {}


def test_collect_runtime_guard_state_with_positions(guard_manager, mock_broker, mock_position):
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]

    state = guard_manager.collect_runtime_guard_state()

    assert len(state.positions) == 1
    assert "BTC-PERP" in state.positions_pnl and "BTC-PERP" in state.positions_dict


def test_collect_runtime_guard_state_uses_broker_pnl(guard_manager, mock_broker, mock_position):
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]
    mock_broker.get_position_pnl.return_value = {
        "realized_pnl": "500",
        "unrealized_pnl": "200",
    }

    state = guard_manager.collect_runtime_guard_state()

    assert state.positions_pnl["BTC-PERP"]["realized_pnl"] == Decimal(
        "500"
    ) and state.positions_pnl["BTC-PERP"]["unrealized_pnl"] == Decimal("200")


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


def test_run_guards_for_state_calls_all_guards(
    guard_manager, sample_guard_state, monkeypatch: pytest.MonkeyPatch
):
    mock_step = MagicMock()
    monkeypatch.setattr(guard_manager, "run_guard_step", mock_step)

    guard_manager.run_guards_for_state(sample_guard_state, incremental=False)

    assert mock_step.call_count == 7
    guard_names = {call[0][0] for call in mock_step.call_args_list}
    assert guard_names == {
        "pnl_telemetry",
        "daily_loss",
        "liquidation_buffer",
        "mark_staleness",
        "risk_metrics",
        "volatility_circuit_breaker",
        "api_health",
    }


def test_run_runtime_guards_first_run(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_state = MagicMock()
    mock_collect = MagicMock(return_value=mock_state)
    mock_run_guards = MagicMock()
    monkeypatch.setattr(guard_manager, "collect_runtime_guard_state", mock_collect)
    monkeypatch.setattr(guard_manager, "run_guards_for_state", mock_run_guards)

    state = guard_manager.run_runtime_guards()

    assert state == mock_state
    mock_collect.assert_called_once()
    mock_run_guards.assert_called_with(mock_state, False)


def test_run_runtime_guards_incremental(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    mock_collect = MagicMock()
    mock_run_guards = MagicMock()
    monkeypatch.setattr(guard_manager, "collect_runtime_guard_state", mock_collect)
    monkeypatch.setattr(guard_manager, "run_guards_for_state", mock_run_guards)

    state = guard_manager.run_runtime_guards()

    assert state == mock_state
    mock_collect.assert_not_called()
    mock_run_guards.assert_called_with(mock_state, True)


def test_run_runtime_guards_force_full(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    new_state = MagicMock()
    mock_collect = MagicMock(return_value=new_state)
    mock_run_guards = MagicMock()
    monkeypatch.setattr(guard_manager, "collect_runtime_guard_state", mock_collect)
    monkeypatch.setattr(guard_manager, "run_guards_for_state", mock_run_guards)

    state = guard_manager.run_runtime_guards(force_full=True)

    assert state == new_state
    mock_collect.assert_called_once()
    mock_run_guards.assert_called_with(new_state, False)


def test_cancel_all_orders_none_cancelled(guard_manager, mock_broker):
    mock_broker.cancel_order.return_value = False

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 0
    guard_manager._invalidate_cache_callback.assert_not_called()


def test_cancel_all_orders_empty_list(mock_broker, mock_risk_manager, mock_equity_calculator):
    gm = GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=mock_equity_calculator,
        open_orders=[],
        invalidate_cache_callback=MagicMock(),
    )

    cancelled_count = gm.cancel_all_orders()

    assert cancelled_count == 0
    mock_broker.cancel_order.assert_not_called()


def test_safe_run_runtime_guards_success(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock()
    monkeypatch.setattr(guard_manager, "run_runtime_guards", mock_run)
    guard_manager.safe_run_runtime_guards()
    mock_run.assert_called_once_with(force_full=False)


def test_safe_run_runtime_guards_force_full(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock()
    monkeypatch.setattr(guard_manager, "run_runtime_guards", mock_run)
    guard_manager.safe_run_runtime_guards(force_full=True)
    mock_run.assert_called_once_with(force_full=True)
