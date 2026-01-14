"""Tests for the health check implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.health_server import HealthState
from gpt_trader.monitoring.health_checks import (
    HealthCheckResult,
    HealthCheckRunner,
    check_broker_ping,
    check_degradation_state,
    check_ws_freshness,
)


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

    def test_high_latency_warning(self) -> None:
        """Test that high latency sets severity to warning."""
        broker = MagicMock()

        # Simulate slow response by mocking time
        with patch("gpt_trader.monitoring.health_checks.time.perf_counter") as mock_time:
            # First call returns 0, second returns 2.5 (2500ms latency)
            mock_time.side_effect = [0, 2.5]

            healthy, details = check_broker_ping(broker)

        assert healthy is True
        assert details["latency_ms"] == 2500.0
        assert details["severity"] == "warning"
        assert "warning" in details


class TestCheckWsFreshness:
    """Tests for check_ws_freshness function."""

    def test_broker_without_ws_support(self) -> None:
        """Test handling when broker doesn't support WS health."""
        broker = MagicMock(spec=["list_balances"])  # No get_ws_health

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details.get("ws_not_supported") is True

    def test_ws_not_initialized(self) -> None:
        """Test handling when WS not initialized (returns None)."""
        broker = MagicMock()
        broker.get_ws_health.return_value = None

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details.get("ws_not_initialized") is True

    def test_ws_connected_and_fresh(self) -> None:
        """Test healthy WS with fresh messages."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 5,  # 5 seconds ago
            "last_heartbeat_ts": time.time() - 10,  # 10 seconds ago
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details["connected"] is True
        assert details["stale"] is False

    def test_ws_stale_message(self) -> None:
        """Test failure when messages are stale."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 120,  # 2 minutes ago (stale)
            "last_heartbeat_ts": time.time() - 30,
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker, message_stale_seconds=60.0)

        assert healthy is False
        assert details["stale"] is True
        assert details["stale_reason"] == "message"

    def test_ws_disconnected(self) -> None:
        """Test failure when WS is disconnected."""
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": False,
            "last_message_ts": 0,
            "last_heartbeat_ts": 0,
            "gap_count": 0,
            "reconnect_count": 5,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["connected"] is False

    def test_ws_max_attempts_triggered(self) -> None:
        """Test critical failure when max reconnect attempts triggered."""
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": False,
            "last_message_ts": 0,
            "last_heartbeat_ts": 0,
            "gap_count": 0,
            "reconnect_count": 10,
            "max_attempts_triggered": True,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["max_attempts_triggered"] is True
        assert details["severity"] == "critical"


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

        # Create a mock with explicit spec to avoid auto-creating attributes
        risk_manager = MagicMock()
        risk_manager.is_reduce_only_mode = MagicMock(return_value=True)
        risk_manager._reduce_only_mode = True
        risk_manager._reduce_only_reason = "validation_failures"
        # Prevent auto-creation of _cfm_reduce_only_mode attribute
        del risk_manager._cfm_reduce_only_mode
        risk_manager.is_cfm_reduce_only_mode = MagicMock(return_value=False)

        healthy, details = check_degradation_state(degradation_state, risk_manager)

        assert healthy is True  # Reduce-only is warning, not failure
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

        assert healthy is True  # Symbol pause is warning, not failure
        assert details["paused_symbol_count"] == 1
        assert "BTC-USD" in details["paused_symbols"]
        assert details["severity"] == "warning"


class TestHealthCheckRunner:
    """Tests for HealthCheckRunner class."""

    def test_initialization(self) -> None:
        """Test runner initialization with default values."""
        health_state = HealthState()
        runner = HealthCheckRunner(health_state=health_state)

        assert runner._health_state is health_state
        assert runner._broker is None
        assert runner._degradation_state is None
        assert runner._risk_manager is None
        assert runner._interval == 30.0
        assert runner._running is False

    def test_set_broker(self) -> None:
        """Test setting broker after initialization."""
        runner = HealthCheckRunner(health_state=HealthState())
        broker = MagicMock()

        runner.set_broker(broker)

        assert runner._broker is broker

    def test_set_degradation_state(self) -> None:
        """Test setting degradation state."""
        runner = HealthCheckRunner(health_state=HealthState())
        state = MagicMock()

        runner.set_degradation_state(state)

        assert runner._degradation_state is state

    def test_run_checks_sync(self) -> None:
        """Test synchronous check execution."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        runner = HealthCheckRunner(
            health_state=HealthState(),
            broker=broker,
            degradation_state=degradation_state,
        )

        results = runner.run_checks_sync()

        assert "broker" in results
        assert "websocket" in results
        assert "degradation" in results
        assert results["broker"][0] is True  # healthy
        assert results["degradation"][0] is True  # healthy

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test async start and stop."""
        import asyncio

        runner = HealthCheckRunner(
            health_state=HealthState(),
            interval_seconds=60.0,
        )  # Long interval to avoid loop execution

        await runner.start()
        assert runner._running is True
        assert runner._task is not None

        # Give the task a moment to start the loop
        await asyncio.sleep(0.01)

        await runner.stop()
        assert runner._running is False
        assert runner._task is None

    @pytest.mark.asyncio
    async def test_execute_checks_updates_health_state(self) -> None:
        """Test that execute_checks updates the global health state."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        health_state = MagicMock()
        runner = HealthCheckRunner(
            health_state=health_state,
            broker=broker,
            interval_seconds=60.0,
        )

        await runner._execute_checks()

        # Verify checks were added
        assert health_state.add_check.call_count >= 2
        call_args = [call[0][0] for call in health_state.add_check.call_args_list]
        assert "broker" in call_args
        assert "websocket" in call_args


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a health check result."""
        result = HealthCheckResult(healthy=True, details={"latency_ms": 50})

        assert result.healthy is True
        assert result.details["latency_ms"] == 50

    def test_unhealthy_result(self) -> None:
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            details={"error": "connection refused", "severity": "critical"},
        )

        assert result.healthy is False
        assert result.details["error"] == "connection refused"


class TestComputeExecutionHealthSignals:
    """Tests for compute_execution_health_signals function."""

    def test_returns_health_summary(self) -> None:
        """Test that function returns a HealthSummary."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthSummary

        result = compute_execution_health_signals()

        assert isinstance(result, HealthSummary)
        assert result.status is not None
        assert len(result.signals) >= 4  # error_rate, retry_rate, latency, guard_trips

    def test_with_custom_thresholds(self) -> None:
        """Test that custom thresholds are respected."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthThresholds

        thresholds = HealthThresholds(
            order_error_rate_warn=0.01,
            order_error_rate_crit=0.05,
        )

        result = compute_execution_health_signals(thresholds=thresholds)

        # Find the order_error_rate signal
        error_rate_signal = next((s for s in result.signals if s.name == "order_error_rate"), None)
        assert error_rate_signal is not None
        assert error_rate_signal.threshold_warn == 0.01
        assert error_rate_signal.threshold_crit == 0.05

    def test_signal_names(self) -> None:
        """Test that all expected signals are present."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals

        result = compute_execution_health_signals()
        signal_names = {s.name for s in result.signals}

        assert "order_error_rate" in signal_names
        assert "order_retry_rate" in signal_names
        assert "broker_latency_p95" in signal_names
        assert "guard_trip_count" in signal_names

    def test_zero_metrics_returns_ok(self) -> None:
        """Test that with no metrics, signals are OK."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus
        from gpt_trader.monitoring.metrics_collector import reset_all

        # Fresh collector should have zero metrics
        reset_all()
        result = compute_execution_health_signals()

        # With zero submissions, error rate should be 0 = OK
        error_signal = next((s for s in result.signals if s.name == "order_error_rate"), None)
        assert error_signal is not None
        assert error_signal.value == 0.0
        assert error_signal.status == HealthStatus.OK

    def test_to_dict_serializable(self) -> None:
        """Test that the summary can be serialized to dict."""
        import json

        from gpt_trader.monitoring.health_checks import compute_execution_health_signals

        result = compute_execution_health_signals()
        result_dict = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None
        assert "status" in result_dict
        assert "signals" in result_dict


class TestCheckWsStalenessSignal:
    """Tests for check_ws_staleness_signal function."""

    def test_broker_without_ws_support(self) -> None:
        """Test handling broker without WS health method."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        broker = MagicMock(spec=["list_balances"])  # No get_ws_health

        signal = check_ws_staleness_signal(broker)

        assert signal.name == "ws_staleness"
        assert signal.status == HealthStatus.UNKNOWN
        assert signal.details.get("ws_not_supported") is True

    def test_ws_not_initialized(self) -> None:
        """Test handling when WS returns None."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        broker = MagicMock()
        broker.get_ws_health.return_value = None

        signal = check_ws_staleness_signal(broker)

        assert signal.name == "ws_staleness"
        assert signal.status == HealthStatus.OK
        assert signal.details.get("ws_not_initialized") is True

    def test_ws_fresh_messages(self) -> None:
        """Test healthy WS with recent messages."""
        import time

        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 5,  # 5 seconds ago
        }

        signal = check_ws_staleness_signal(broker)

        assert signal.name == "ws_staleness"
        assert signal.status == HealthStatus.OK
        assert signal.value < 10  # Should be close to 5 seconds

    def test_ws_stale_messages_warn(self) -> None:
        """Test warning when messages are moderately stale."""
        import time

        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 45,  # 45 seconds ago
        }

        thresholds = HealthThresholds(
            ws_staleness_seconds_warn=30.0,
            ws_staleness_seconds_crit=60.0,
        )
        signal = check_ws_staleness_signal(broker, thresholds=thresholds)

        assert signal.status == HealthStatus.WARN
        assert 40 < signal.value < 50

    def test_ws_stale_messages_crit(self) -> None:
        """Test critical when messages are very stale."""
        import time

        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 120,  # 2 minutes ago
        }

        thresholds = HealthThresholds(
            ws_staleness_seconds_warn=30.0,
            ws_staleness_seconds_crit=60.0,
        )
        signal = check_ws_staleness_signal(broker, thresholds=thresholds)

        assert signal.status == HealthStatus.CRIT
        assert signal.value > 100

    def test_ws_exception_returns_unknown(self) -> None:
        """Test that exceptions result in UNKNOWN status."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        broker = MagicMock()
        broker.get_ws_health.side_effect = RuntimeError("Connection error")

        signal = check_ws_staleness_signal(broker)

        assert signal.status == HealthStatus.UNKNOWN
        assert "error" in signal.details
