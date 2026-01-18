"""Tests for mark staleness and risk metrics guards."""

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardComputationError,
    RiskGuardDataUnavailable,
    RiskGuardTelemetryError,
)


def test_guard_mark_staleness_no_cache(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    del mock_broker._mark_cache

    guard_manager.guard_mark_staleness(sample_guard_state)
    mock_risk_manager.check_mark_staleness.assert_not_called()


def test_guard_mark_staleness_with_cache(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    mock_broker._mark_cache = MagicMock()
    mock_broker._mark_cache.get_mark.return_value = None
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}

    guard_manager.guard_mark_staleness(sample_guard_state)

    mock_risk_manager.check_mark_staleness.assert_called_with("BTC-PERP")


def test_guard_mark_staleness_fetch_failure(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    mock_broker._mark_cache = MagicMock()
    mock_broker._mark_cache.get_mark.side_effect = Exception("Cache error")
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}

    with pytest.raises(RiskGuardDataUnavailable):
        guard_manager.guard_mark_staleness(sample_guard_state)


def test_guard_risk_metrics_success(guard_manager, sample_guard_state, mock_risk_manager):
    guard_manager.guard_risk_metrics(sample_guard_state)

    mock_risk_manager.append_risk_metrics.assert_called_once()


def test_guard_risk_metrics_failure(guard_manager, sample_guard_state, mock_risk_manager):
    mock_risk_manager.append_risk_metrics.side_effect = Exception("Metrics error")

    with pytest.raises(RiskGuardTelemetryError):
        guard_manager.guard_risk_metrics(sample_guard_state)


def test_guard_risk_metrics_propagates_guard_error(
    guard_manager, sample_guard_state, mock_risk_manager
):
    error = RiskGuardComputationError(guard_name="risk_metrics", message="Test", details={})
    mock_risk_manager.append_risk_metrics.side_effect = error

    with pytest.raises(RiskGuardComputationError):
        guard_manager.guard_risk_metrics(sample_guard_state)
