"""Tests for execution health signals and websocket staleness signals."""

from __future__ import annotations

from unittest.mock import MagicMock


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
