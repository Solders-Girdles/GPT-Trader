"""Tests for HealthThresholds."""

from __future__ import annotations

from gpt_trader.monitoring.health_signals import HealthThresholds


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
