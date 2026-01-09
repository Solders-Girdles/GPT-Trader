"""Tests for health signal models."""

from __future__ import annotations

from gpt_trader.monitoring.health_signals import (
    HealthSignal,
    HealthStatus,
    HealthSummary,
    HealthThresholds,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_enum_values(self) -> None:
        assert HealthStatus.OK.value == "OK"
        assert HealthStatus.WARN.value == "WARN"
        assert HealthStatus.CRIT.value == "CRIT"
        assert HealthStatus.UNKNOWN.value == "UNKNOWN"

    def test_string_inheritance(self) -> None:
        # HealthStatus inherits from str, so .value gives the string
        assert HealthStatus.OK.value == "OK"
        # When used in f-strings, use .value for the string representation
        assert f"Status: {HealthStatus.WARN.value}" == "Status: WARN"


class TestHealthSignal:
    """Tests for HealthSignal dataclass."""

    def test_creation(self) -> None:
        signal = HealthSignal(
            name="order_error_rate",
            status=HealthStatus.OK,
            value=0.02,
            threshold_warn=0.05,
            threshold_crit=0.15,
            unit="%",
        )
        assert signal.name == "order_error_rate"
        assert signal.status == HealthStatus.OK
        assert signal.value == 0.02
        assert signal.threshold_warn == 0.05
        assert signal.threshold_crit == 0.15
        assert signal.unit == "%"
        assert signal.details == {}

    def test_from_value_ok(self) -> None:
        signal = HealthSignal.from_value(
            name="error_rate",
            value=0.01,
            threshold_warn=0.05,
            threshold_crit=0.15,
        )
        assert signal.status == HealthStatus.OK
        assert signal.value == 0.01

    def test_from_value_warn(self) -> None:
        signal = HealthSignal.from_value(
            name="error_rate",
            value=0.08,
            threshold_warn=0.05,
            threshold_crit=0.15,
        )
        assert signal.status == HealthStatus.WARN
        assert signal.value == 0.08

    def test_from_value_crit(self) -> None:
        signal = HealthSignal.from_value(
            name="error_rate",
            value=0.20,
            threshold_warn=0.05,
            threshold_crit=0.15,
        )
        assert signal.status == HealthStatus.CRIT
        assert signal.value == 0.20

    def test_from_value_at_warn_threshold(self) -> None:
        """Test that value exactly at warn threshold triggers WARN."""
        signal = HealthSignal.from_value(
            name="error_rate",
            value=0.05,
            threshold_warn=0.05,
            threshold_crit=0.15,
        )
        assert signal.status == HealthStatus.WARN

    def test_from_value_at_crit_threshold(self) -> None:
        """Test that value exactly at crit threshold triggers CRIT."""
        signal = HealthSignal.from_value(
            name="error_rate",
            value=0.15,
            threshold_warn=0.05,
            threshold_crit=0.15,
        )
        assert signal.status == HealthStatus.CRIT

    def test_from_value_lower_is_worse(self) -> None:
        """Test availability-style metrics where lower is worse."""
        # High availability is good
        signal_ok = HealthSignal.from_value(
            name="availability",
            value=0.999,
            threshold_warn=0.99,
            threshold_crit=0.95,
            higher_is_worse=False,
        )
        assert signal_ok.status == HealthStatus.OK

        # Low availability is warning
        signal_warn = HealthSignal.from_value(
            name="availability",
            value=0.98,
            threshold_warn=0.99,
            threshold_crit=0.95,
            higher_is_worse=False,
        )
        assert signal_warn.status == HealthStatus.WARN

        # Very low availability is critical
        signal_crit = HealthSignal.from_value(
            name="availability",
            value=0.90,
            threshold_warn=0.99,
            threshold_crit=0.95,
            higher_is_worse=False,
        )
        assert signal_crit.status == HealthStatus.CRIT

    def test_from_value_with_details(self) -> None:
        signal = HealthSignal.from_value(
            name="latency_p95",
            value=1500.0,
            threshold_warn=1000.0,
            threshold_crit=3000.0,
            unit="ms",
            details={"endpoint": "/orders"},
        )
        assert signal.details == {"endpoint": "/orders"}
        assert signal.unit == "ms"

    def test_to_dict(self) -> None:
        signal = HealthSignal(
            name="order_error_rate",
            status=HealthStatus.WARN,
            value=0.08,
            threshold_warn=0.05,
            threshold_crit=0.15,
            unit="%",
            details={"window_seconds": 300},
        )
        result = signal.to_dict()

        assert result["name"] == "order_error_rate"
        assert result["status"] == "WARN"
        assert result["value"] == 0.08
        assert result["threshold_warn"] == 0.05
        assert result["threshold_crit"] == 0.15
        assert result["unit"] == "%"
        assert result["details"]["window_seconds"] == 300


class TestHealthSummary:
    """Tests for HealthSummary dataclass."""

    def test_from_signals_empty(self) -> None:
        summary = HealthSummary.from_signals([])
        assert summary.status == HealthStatus.UNKNOWN
        assert summary.signals == []
        assert "No health signals available" in summary.message

    def test_from_signals_all_ok(self) -> None:
        signals = [
            HealthSignal.from_value("error_rate", 0.01, 0.05, 0.15),
            HealthSignal.from_value("latency", 500, 1000, 3000),
        ]
        summary = HealthSummary.from_signals(signals)

        assert summary.status == HealthStatus.OK
        assert len(summary.signals) == 2
        assert "All signals OK" in summary.message

    def test_from_signals_worst_status_wins(self) -> None:
        signals = [
            HealthSignal.from_value("error_rate", 0.01, 0.05, 0.15),  # OK
            HealthSignal.from_value("latency", 1500, 1000, 3000),  # WARN
        ]
        summary = HealthSummary.from_signals(signals)

        assert summary.status == HealthStatus.WARN
        assert "1 warning" in summary.message

    def test_from_signals_crit_takes_precedence(self) -> None:
        signals = [
            HealthSignal.from_value("error_rate", 0.20, 0.05, 0.15),  # CRIT
            HealthSignal.from_value("latency", 1500, 1000, 3000),  # WARN
        ]
        summary = HealthSummary.from_signals(signals)

        assert summary.status == HealthStatus.CRIT
        assert "1 critical" in summary.message

    def test_from_signals_multiple_crit(self) -> None:
        signals = [
            HealthSignal.from_value("error_rate", 0.20, 0.05, 0.15),  # CRIT
            HealthSignal.from_value("latency", 4000, 1000, 3000),  # CRIT
        ]
        summary = HealthSummary.from_signals(signals)

        assert summary.status == HealthStatus.CRIT
        assert "2 critical" in summary.message

    def test_to_dict(self) -> None:
        signals = [
            HealthSignal.from_value("error_rate", 0.08, 0.05, 0.15),
        ]
        summary = HealthSummary.from_signals(signals)
        result = summary.to_dict()

        assert result["status"] == "WARN"
        assert len(result["signals"]) == 1
        assert result["signals"][0]["name"] == "error_rate"
        assert "warning" in result["message"]


class TestHealthThresholds:
    """Tests for HealthThresholds dataclass."""

    def test_default_values(self) -> None:
        thresholds = HealthThresholds()

        assert thresholds.order_error_rate_warn == 0.05
        assert thresholds.order_error_rate_crit == 0.15
        assert thresholds.order_retry_rate_warn == 0.10
        assert thresholds.order_retry_rate_crit == 0.25
        assert thresholds.broker_latency_ms_warn == 1000.0
        assert thresholds.broker_latency_ms_crit == 3000.0
        assert thresholds.ws_staleness_seconds_warn == 30.0
        assert thresholds.ws_staleness_seconds_crit == 60.0
        assert thresholds.guard_trip_count_warn == 3
        assert thresholds.guard_trip_count_crit == 10

    def test_custom_values(self) -> None:
        thresholds = HealthThresholds(
            order_error_rate_warn=0.02,
            order_error_rate_crit=0.10,
        )
        assert thresholds.order_error_rate_warn == 0.02
        assert thresholds.order_error_rate_crit == 0.10
        # Others retain defaults
        assert thresholds.broker_latency_ms_warn == 1000.0
