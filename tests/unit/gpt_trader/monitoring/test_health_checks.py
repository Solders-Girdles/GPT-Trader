"""Tests for broker ping and degradation state health checks."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.health_checks import check_broker_ping, check_degradation_state


class TestCheckBrokerPing:
    """Tests for check_broker_ping function."""

    def test_success_with_get_time(self) -> None:
        """Test successful ping using get_time method."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 1234567890}
        healthy, details = check_broker_ping(broker)
        assert healthy is True
        assert "latency_ms" in details
        assert details["method"] == "get_time"
        assert details["severity"] == "critical"
        broker.get_time.assert_called_once()

    def test_success_fallback_to_list_balances(self) -> None:
        """Test fallback to list_balances when get_time not available."""
        broker = MagicMock(spec=["list_balances"])
        broker.list_balances.return_value = [{"currency": "USD", "available": "100"}]
        healthy, details = check_broker_ping(broker)
        assert healthy is True
        assert details["method"] == "list_balances"
        broker.list_balances.assert_called_once()

    def test_failure_on_exception(self) -> None:
        """Test failure when broker call raises exception."""
        broker = MagicMock()
        broker.get_time.side_effect = ConnectionError("connection refused")
        healthy, details = check_broker_ping(broker)
        assert healthy is False
        assert "error" in details
        assert details["error_type"] == "ConnectionError"
        assert details["severity"] == "critical"

    def test_high_latency_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that high latency sets severity to warning."""
        broker = MagicMock()
        mock_time = MagicMock()
        mock_time.side_effect = [0, 2.5]
        monkeypatch.setattr(time, "perf_counter", mock_time)
        healthy, details = check_broker_ping(broker)
        assert healthy is True
        assert details["latency_ms"] == 2500.0
        assert details["severity"] == "warning"
        assert "warning" in details


class TestCheckDegradationState:
    """Tests for check_degradation_state function."""

    def test_normal_operation(self) -> None:
        """Test healthy state when no degradation."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }
        healthy, details = check_degradation_state(degradation_state)
        assert healthy is True
        assert details["global_paused"] is False
        assert details["reduce_only_mode"] is False

    def test_global_paused(self) -> None:
        """Test failure when globally paused."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": True,
            "global_reason": "max_reconnect_attempts",
            "paused_symbols": {},
            "global_remaining_seconds": 300,
        }
        healthy, details = check_degradation_state(degradation_state)
        assert healthy is False
        assert details["global_paused"] is True
        assert details["severity"] == "critical"

    def test_reduce_only_mode(self) -> None:
        """Test warning when in reduce-only mode."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }
        risk_manager = MagicMock()
        risk_manager.is_reduce_only_mode = MagicMock(return_value=True)
        risk_manager._reduce_only_mode = True
        risk_manager._reduce_only_reason = "validation_failures"
        del risk_manager._cfm_reduce_only_mode
        risk_manager.is_cfm_reduce_only_mode = MagicMock(return_value=False)
        healthy, details = check_degradation_state(degradation_state, risk_manager)
        assert healthy is True
        assert details["reduce_only_mode"] is True
        assert details["reduce_only_reason"] == "validation_failures"
        assert details["severity"] == "warning"

    def test_symbol_paused(self) -> None:
        """Test warning when specific symbols are paused."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {"BTC-USD": {"reason": "rate_limited"}},
            "global_remaining_seconds": 0,
        }
        healthy, details = check_degradation_state(degradation_state)
        assert healthy is True
        assert details["paused_symbol_count"] == 1
        assert "BTC-USD" in details["paused_symbols"]
        assert details["severity"] == "warning"
