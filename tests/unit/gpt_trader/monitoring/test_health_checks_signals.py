"""Tests for execution health signals and websocket staleness signals."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.metrics_collector import reset_all
from gpt_trader.utilities.time_provider import FakeClock


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    """Reset metrics before and after each test for deterministic results."""
    reset_all()
    yield
    reset_all()


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
        assert "missing_decision_id_count" in signal_names

    def test_zero_metrics_returns_ok(self) -> None:
        """Test that with no metrics, signals are OK."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus

        result = compute_execution_health_signals()

        assert result.status == HealthStatus.OK
        assert all(signal.status == HealthStatus.OK for signal in result.signals)

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

    def test_guard_trip_warn_at_threshold(self) -> None:
        """Test that value exactly at warn threshold returns WARN."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus
        from gpt_trader.monitoring.metrics_collector import record_counter

        record_counter("gpt_trader_guard_trip_count", increment=3)

        result = compute_execution_health_signals()
        guard_signal = next((s for s in result.signals if s.name == "guard_trip_count"), None)

        assert guard_signal is not None
        assert guard_signal.value == pytest.approx(3.0)
        assert guard_signal.status == HealthStatus.WARN

    def test_guard_trip_crit_at_threshold(self) -> None:
        """Test that value exactly at crit threshold returns CRIT."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus
        from gpt_trader.monitoring.metrics_collector import record_counter

        record_counter("gpt_trader_guard_trip_count", increment=10)

        result = compute_execution_health_signals()
        guard_signal = next((s for s in result.signals if s.name == "guard_trip_count"), None)

        assert guard_signal is not None
        assert guard_signal.value == pytest.approx(10.0)
        assert guard_signal.status == HealthStatus.CRIT

    def test_missing_decision_id_warn_at_threshold(self) -> None:
        """Test that missing decision_id count at warn threshold returns WARN."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus
        from gpt_trader.monitoring.metrics_collector import record_counter

        record_counter("gpt_trader_order_missing_decision_id_total", increment=1)

        result = compute_execution_health_signals()
        missing_signal = next(
            (s for s in result.signals if s.name == "missing_decision_id_count"), None
        )

        assert missing_signal is not None
        assert missing_signal.value == pytest.approx(1.0)
        assert missing_signal.status == HealthStatus.WARN

    def test_custom_thresholds_override_status(self) -> None:
        """Test that custom thresholds impact computed status."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds
        from gpt_trader.monitoring.metrics_collector import record_histogram

        record_histogram(
            "gpt_trader_broker_call_latency_seconds",
            1.0,
            labels={"outcome": "success"},
        )

        thresholds = HealthThresholds(
            broker_latency_ms_warn=1500.0,
            broker_latency_ms_crit=2500.0,
        )

        result = compute_execution_health_signals(thresholds=thresholds)
        latency_signal = next((s for s in result.signals if s.name == "broker_latency_p95"), None)

        assert latency_signal is not None
        assert latency_signal.value == pytest.approx(1000.0)
        assert latency_signal.status == HealthStatus.OK

    def test_missing_metrics_returns_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing metrics data yields OK signals."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus

        class StubCollector:
            def get_metrics_summary(self) -> dict[str, object]:
                return {"counters": None, "histograms": None}

        monkeypatch.setattr(
            "gpt_trader.monitoring.metrics_collector.get_metrics_collector",
            lambda: StubCollector(),
        )

        result = compute_execution_health_signals()

        assert result.status == HealthStatus.OK
        assert all(signal.status == HealthStatus.OK for signal in result.signals)

    def test_mixed_statuses_aggregate_summary(self) -> None:
        """Test aggregate status with mixed signal levels."""
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals
        from gpt_trader.monitoring.health_signals import HealthStatus
        from gpt_trader.monitoring.metrics_collector import record_counter, record_histogram

        record_counter(
            "gpt_trader_order_submission_total",
            increment=8,
            labels={"result": "success"},
        )
        record_counter(
            "gpt_trader_order_submission_total",
            increment=2,
            labels={"result": "failed"},
        )
        for _ in range(11):
            record_histogram(
                "gpt_trader_broker_call_latency_seconds",
                0.01,
                labels={"outcome": "success"},
            )

        result = compute_execution_health_signals()

        assert result.status == HealthStatus.CRIT
        assert result.message == "1 critical signal(s), 1 warning(s)"


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
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        clock = FakeClock(start_time=1000.0)
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": 995.0,  # 5 seconds ago
        }

        signal = check_ws_staleness_signal(broker, time_provider=clock)

        assert signal.name == "ws_staleness"
        assert signal.status == HealthStatus.OK
        assert signal.value == pytest.approx(5.0)

    def test_ws_stale_messages_warn(self) -> None:
        """Test warning when messages are moderately stale."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds

        clock = FakeClock(start_time=1000.0)
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": 955.0,  # 45 seconds ago
        }

        thresholds = HealthThresholds(
            ws_staleness_seconds_warn=30.0,
            ws_staleness_seconds_crit=60.0,
        )
        signal = check_ws_staleness_signal(
            broker,
            thresholds=thresholds,
            time_provider=clock,
        )

        assert signal.status == HealthStatus.WARN
        assert signal.value == pytest.approx(45.0)

    def test_ws_stale_messages_crit(self) -> None:
        """Test critical when messages are very stale."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds

        clock = FakeClock(start_time=1000.0)
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": 880.0,  # 2 minutes ago
        }

        thresholds = HealthThresholds(
            ws_staleness_seconds_warn=30.0,
            ws_staleness_seconds_crit=60.0,
        )
        signal = check_ws_staleness_signal(
            broker,
            thresholds=thresholds,
            time_provider=clock,
        )

        assert signal.status == HealthStatus.CRIT
        assert signal.value == pytest.approx(120.0)

    @pytest.mark.parametrize(
        ("age_seconds", "expected_status"),
        [
            (29.9, "OK"),
            (30.0, "WARN"),
            (60.1, "CRIT"),
        ],
    )
    def test_ws_staleness_signal_threshold_boundaries(
        self,
        age_seconds: float,
        expected_status: str,
    ) -> None:
        """Test staleness signal transitions around threshold boundaries."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus, HealthThresholds

        clock = FakeClock(start_time=1000.0)
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": clock.time() - age_seconds,
        }

        thresholds = HealthThresholds(
            ws_staleness_seconds_warn=30.0,
            ws_staleness_seconds_crit=60.0,
        )
        signal = check_ws_staleness_signal(
            broker,
            thresholds=thresholds,
            time_provider=clock,
        )

        assert signal.status == HealthStatus(expected_status)
        assert signal.value == pytest.approx(age_seconds)

    def test_ws_exception_returns_unknown(self) -> None:
        """Test that exceptions result in UNKNOWN status."""
        from gpt_trader.monitoring.health_checks import check_ws_staleness_signal
        from gpt_trader.monitoring.health_signals import HealthStatus

        broker = MagicMock()
        broker.get_ws_health.side_effect = RuntimeError("Connection error")

        signal = check_ws_staleness_signal(broker)

        assert signal.status == HealthStatus.UNKNOWN
        assert "error" in signal.details
