"""Tests for GuardManager guard-step execution and telemetry logging."""

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardTelemetryError,
)


def test_run_guard_step_success(guard_manager):
    func = MagicMock()

    with patch(
        "gpt_trader.features.live_trade.execution.guard_manager.record_guard_success"
    ) as mock_success:
        guard_manager.run_guard_step("test_guard", func)

    func.assert_called_once()
    mock_success.assert_called_with("test_guard")


def test_run_guard_step_recoverable_error(guard_manager):
    error = RiskGuardTelemetryError(
        guard_name="test_guard",
        message="Recoverable error",
        details={},
    )
    func = MagicMock(side_effect=error)

    with patch(
        "gpt_trader.features.live_trade.execution.guard_manager.record_guard_failure"
    ) as mock_failure:
        guard_manager.run_guard_step("test_guard", func)

    mock_failure.assert_called_once()


def test_run_guard_step_unrecoverable_error(guard_manager):
    error = RiskGuardActionError(
        guard_name="test_guard",
        message="Fatal error",
        details={},
    )
    func = MagicMock(side_effect=error)

    with patch("gpt_trader.features.live_trade.execution.guard_manager.record_guard_failure"):
        with pytest.raises(RiskGuardActionError):
            guard_manager.run_guard_step("test_guard", func)


def test_run_guard_step_unexpected_error(guard_manager):
    func = MagicMock(side_effect=ValueError("Unexpected"))

    with patch(
        "gpt_trader.features.live_trade.execution.guard_manager.record_guard_failure"
    ) as mock_failure:
        with pytest.raises(RiskGuardComputationError):
            guard_manager.run_guard_step("test_guard", func)

    assert mock_failure.called


def test_log_guard_telemetry_success(guard_manager, sample_guard_state):
    with patch(
        "gpt_trader.features.live_trade.execution.guards.pnl_telemetry._get_plog"
    ) as mock_get_plog:
        mock_plog = MagicMock()
        mock_get_plog.return_value = mock_plog

        guard_manager.log_guard_telemetry(sample_guard_state)

        mock_plog.log_pnl.assert_called_once()


def test_log_guard_telemetry_failure_raises(guard_manager, sample_guard_state):
    with patch(
        "gpt_trader.features.live_trade.execution.guards.pnl_telemetry._get_plog"
    ) as mock_get_plog:
        mock_plog = MagicMock()
        mock_plog.log_pnl.side_effect = Exception("Telemetry failed")
        mock_get_plog.return_value = mock_plog

        with pytest.raises(RiskGuardTelemetryError) as exc_info:
            guard_manager.log_guard_telemetry(sample_guard_state)

        assert "BTC-PERP" in str(exc_info.value.details)
