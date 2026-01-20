"""Tests for GuardManager guard-step execution and telemetry logging."""

from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.guard_manager as guard_manager_module
import gpt_trader.features.live_trade.execution.guards.pnl_telemetry as pnl_telemetry_module
from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardTelemetryError,
)


@pytest.fixture
def record_guard_success_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_success = MagicMock()
    monkeypatch.setattr(guard_manager_module, "record_guard_success", mock_success)
    return mock_success


@pytest.fixture
def record_guard_failure_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_failure = MagicMock()
    monkeypatch.setattr(guard_manager_module, "record_guard_failure", mock_failure)
    return mock_failure


@pytest.fixture
def plog_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_plog = MagicMock()
    monkeypatch.setattr(pnl_telemetry_module, "_get_plog", lambda: mock_plog)
    return mock_plog


def test_run_guard_step_success(guard_manager, record_guard_success_mock):
    func = MagicMock()

    guard_manager.run_guard_step("test_guard", func)

    func.assert_called_once()
    record_guard_success_mock.assert_called_with("test_guard")


def test_run_guard_step_recoverable_error(guard_manager, record_guard_failure_mock):
    error = RiskGuardTelemetryError(
        guard_name="test_guard",
        message="Recoverable error",
        details={},
    )
    func = MagicMock(side_effect=error)

    guard_manager.run_guard_step("test_guard", func)

    record_guard_failure_mock.assert_called_once()


def test_run_guard_step_unrecoverable_error(guard_manager, record_guard_failure_mock):
    error = RiskGuardActionError(
        guard_name="test_guard",
        message="Fatal error",
        details={},
    )
    func = MagicMock(side_effect=error)

    with pytest.raises(RiskGuardActionError):
        guard_manager.run_guard_step("test_guard", func)


def test_run_guard_step_unexpected_error(guard_manager, record_guard_failure_mock):
    func = MagicMock(side_effect=ValueError("Unexpected"))

    with pytest.raises(RiskGuardComputationError):
        guard_manager.run_guard_step("test_guard", func)

    assert record_guard_failure_mock.called


def test_log_guard_telemetry_success(guard_manager, sample_guard_state, plog_mock):
    guard_manager.log_guard_telemetry(sample_guard_state)

    plog_mock.log_pnl.assert_called_once()


def test_log_guard_telemetry_failure_raises(guard_manager, sample_guard_state, plog_mock):
    plog_mock.log_pnl.side_effect = Exception("Telemetry failed")

    with pytest.raises(RiskGuardTelemetryError) as exc_info:
        guard_manager.log_guard_telemetry(sample_guard_state)

    assert "BTC-PERP" in str(exc_info.value.details)
