"""Tests for GuardManager orchestration across guard phases."""

import time
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardDataUnavailable,
)


def test_run_guards_for_state_calls_all_guards(guard_manager, sample_guard_state):
    with patch.object(guard_manager, "run_guard_step") as mock_step:
        guard_manager.run_guards_for_state(sample_guard_state, incremental=False)

    assert mock_step.call_count == 7
    guard_names = [call[0][0] for call in mock_step.call_args_list]
    assert "pnl_telemetry" in guard_names
    assert "daily_loss" in guard_names
    assert "liquidation_buffer" in guard_names
    assert "mark_staleness" in guard_names
    assert "risk_metrics" in guard_names
    assert "volatility_circuit_breaker" in guard_names
    assert "api_health" in guard_names


def test_run_runtime_guards_first_run(guard_manager):
    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        mock_state = MagicMock()
        mock_collect.return_value = mock_state

        state = guard_manager.run_runtime_guards()

        assert state == mock_state
        mock_collect.assert_called_once()
        mock_run_guards.assert_called_with(mock_state, False)


def test_run_runtime_guards_incremental(guard_manager):
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        state = guard_manager.run_runtime_guards()

        assert state == mock_state
        mock_collect.assert_not_called()
        mock_run_guards.assert_called_with(mock_state, True)


def test_run_runtime_guards_force_full(guard_manager):
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        new_state = MagicMock()
        mock_collect.return_value = new_state

        state = guard_manager.run_runtime_guards(force_full=True)

        assert state == new_state
        mock_collect.assert_called_once()
        mock_run_guards.assert_called_with(new_state, False)


class TestGuardManagerEdgeCases:
    def test_data_unavailable_is_recorded_and_surfaces(self, guard_manager, sample_guard_state):
        error = RiskGuardDataUnavailable(
            guard_name="test_guard",
            message="Data temporarily unavailable",
            details={"reason": "network"},
        )
        func = MagicMock(side_effect=error)

        with patch(
            "gpt_trader.features.live_trade.execution.guard_manager.record_guard_failure"
        ) as mock_record:
            guard_manager.run_guard_step("test_guard", func)

        mock_record.assert_called_once()
        recorded_error = mock_record.call_args[0][0]
        assert recorded_error.guard_name == "test_guard"
        assert recorded_error.recoverable is True

    def test_guard_order_is_stable(self, guard_manager):
        expected_order = [
            "pnl_telemetry",
            "daily_loss",
            "liquidation_buffer",
            "mark_staleness",
            "risk_metrics",
            "volatility_circuit_breaker",
            "api_health",
        ]

        actual_order = [guard.name for guard in guard_manager._guards]
        assert actual_order == expected_order

    def test_run_guards_for_state_stops_on_unrecoverable_error(
        self, guard_manager, sample_guard_state
    ):
        call_order: list[str] = []

        def make_check(name: str, should_fail: bool = False):
            def check(state, incremental: bool = False):
                call_order.append(name)
                if should_fail:
                    raise RiskGuardActionError(
                        guard_name=name,
                        message="Fatal",
                        details={},
                    )

            return check

        guard_manager._guards = []
        for i, name in enumerate(["guard_a", "guard_b", "guard_c"]):
            mock_guard = MagicMock()
            mock_guard.name = name
            mock_guard.check = make_check(name, should_fail=(i == 1))
            guard_manager._guards.append(mock_guard)

        with pytest.raises(RiskGuardActionError):
            guard_manager.run_guards_for_state(sample_guard_state, incremental=False)

        assert call_order == ["guard_a", "guard_b"]
