"""Tests for daily report risk analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import calculate_risk_metrics


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_risk_metrics([])
        assert result["guard_triggers"] == {}
        assert result["circuit_breaker_state"] == {}

    def test_counts_guard_triggers(self) -> None:
        events = [
            {"type": "guard_triggered", "guard": "daily_loss"},
            {"type": "guard_triggered", "guard": "daily_loss"},
            {"type": "guard_triggered", "guard": "volatility"},
        ]
        result = calculate_risk_metrics(events)
        assert result["guard_triggers"]["daily_loss"] == 2
        assert result["guard_triggers"]["volatility"] == 1

    def test_handles_unknown_guard(self) -> None:
        events = [{"type": "guard_triggered"}]
        result = calculate_risk_metrics(events)
        assert result["guard_triggers"]["unknown"] == 1

    def test_tracks_circuit_breaker_state(self) -> None:
        events = [
            {
                "type": "circuit_breaker_triggered",
                "rule": "max_drawdown",
                "action": "halt_trading",
                "timestamp": "2024-01-01T12:00:00",
            },
        ]
        result = calculate_risk_metrics(events)
        assert result["circuit_breaker_state"]["triggered"] is True
        assert result["circuit_breaker_state"]["rule"] == "max_drawdown"
        assert result["circuit_breaker_state"]["action"] == "halt_trading"
