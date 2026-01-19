"""Tests for daily report health analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import calculate_health_metrics


class TestCalculateHealthMetrics:
    """Tests for calculate_health_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_health_metrics([])
        assert result["stale_marks_count"] == 0
        assert result["ws_reconnects"] == 0
        assert result["unfilled_orders"] == 0
        assert result["api_errors"] == 0

    def test_counts_stale_marks(self) -> None:
        events = [
            {"type": "stale_mark_detected"},
            {"type": "stale_mark_detected"},
        ]
        result = calculate_health_metrics(events)
        assert result["stale_marks_count"] == 2

    def test_counts_ws_reconnects(self) -> None:
        events = [
            {"type": "websocket_reconnect"},
            {"type": "websocket_reconnect"},
            {"type": "websocket_reconnect"},
        ]
        result = calculate_health_metrics(events)
        assert result["ws_reconnects"] == 3

    def test_counts_unfilled_orders(self) -> None:
        events = [{"type": "unfilled_order_alert"}]
        result = calculate_health_metrics(events)
        assert result["unfilled_orders"] == 1

    def test_counts_api_errors(self) -> None:
        events = [
            {"type": "api_error"},
            {"type": "api_error"},
        ]
        result = calculate_health_metrics(events)
        assert result["api_errors"] == 2

    def test_counts_all_types(self) -> None:
        events = [
            {"type": "stale_mark_detected"},
            {"type": "websocket_reconnect"},
            {"type": "unfilled_order_alert"},
            {"type": "api_error"},
        ]
        result = calculate_health_metrics(events)
        assert result["stale_marks_count"] == 1
        assert result["ws_reconnects"] == 1
        assert result["unfilled_orders"] == 1
        assert result["api_errors"] == 1
