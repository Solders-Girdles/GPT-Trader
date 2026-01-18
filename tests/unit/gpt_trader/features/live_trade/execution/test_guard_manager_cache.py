"""Tests for GuardManager cache invalidation and full-guard scheduling."""

import time
from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.guard_manager import GuardManager


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
