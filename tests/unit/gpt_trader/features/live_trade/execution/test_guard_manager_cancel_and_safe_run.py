"""Tests for GuardManager order cancellation and safe-run wrapper."""

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardTelemetryError,
)


def test_cancel_all_orders_success(guard_manager, mock_broker):
    mock_broker.cancel_order.return_value = True

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 2
    assert mock_broker.cancel_order.call_count == 2
    assert len(guard_manager.open_orders) == 0
    guard_manager._invalidate_cache_callback.assert_called()


def test_cancel_all_orders_partial_failure(guard_manager, mock_broker):
    mock_broker.cancel_order.side_effect = [Exception("Failed"), True]

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 1
    assert "order1" in guard_manager.open_orders
    assert "order2" not in guard_manager.open_orders


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


def test_cancel_all_orders_uses_broker_list_orders(guard_manager, mock_broker):
    mock_broker.list_orders = MagicMock(
        return_value={"orders": [{"id": "order3"}, {"id": "order4"}]}
    )
    mock_broker.cancel_order.return_value = True

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 2
    cancelled_ids = [call_args[0][0] for call_args in mock_broker.cancel_order.call_args_list]
    assert cancelled_ids == ["order3", "order4"]
    assert guard_manager.open_orders == []


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


def test_safe_run_runtime_guards_recoverable_error(
    guard_manager, mock_risk_manager, monkeypatch: pytest.MonkeyPatch
):
    error = RiskGuardTelemetryError(guard_name="test", message="Recoverable", details={})

    monkeypatch.setattr(guard_manager, "run_runtime_guards", MagicMock(side_effect=error))
    guard_manager.safe_run_runtime_guards()

    mock_risk_manager.set_reduce_only_mode.assert_not_called()


def test_safe_run_runtime_guards_unrecoverable_error(
    guard_manager, mock_risk_manager, monkeypatch: pytest.MonkeyPatch
):
    error = RiskGuardActionError(guard_name="test", message="Fatal", details={})

    monkeypatch.setattr(guard_manager, "run_runtime_guards", MagicMock(side_effect=error))
    guard_manager.safe_run_runtime_guards()

    mock_risk_manager.set_reduce_only_mode.assert_called_with(True, reason="guard_failure")
    guard_manager._invalidate_cache_callback.assert_called()


def test_safe_run_runtime_guards_unexpected_error(guard_manager, monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock(side_effect=ValueError("Unexpected"))
    monkeypatch.setattr(guard_manager, "run_runtime_guards", mock_run)
    guard_manager.safe_run_runtime_guards()
    mock_run.assert_called_once_with(force_full=False)


def test_safe_run_runtime_guards_reduce_only_failure(
    guard_manager, mock_risk_manager, monkeypatch: pytest.MonkeyPatch
):
    error = RiskGuardActionError(guard_name="test", message="Fatal", details={})
    mock_risk_manager.set_reduce_only_mode.side_effect = Exception("Failed")

    monkeypatch.setattr(guard_manager, "run_runtime_guards", MagicMock(side_effect=error))
    guard_manager.safe_run_runtime_guards()

    guard_manager._invalidate_cache_callback.assert_called()
