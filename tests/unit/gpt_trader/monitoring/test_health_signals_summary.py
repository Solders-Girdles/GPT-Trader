"""Tests for HealthSummary."""

from __future__ import annotations

from gpt_trader.monitoring.health_signals import (
    HealthSignal,
    HealthStatus,
    HealthSummary,
)


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
