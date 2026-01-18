"""Unit tests for status reporter metrics recording."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter


class TestStatusReporterMetrics:
    """Tests for metrics recording in StatusReporter."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before and after each test."""
        from gpt_trader.monitoring.metrics_collector import reset_all

        reset_all()
        yield
        reset_all()

    def test_update_equity_records_gauge(self) -> None:
        """update_equity records the equity gauge metric."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_equity(Decimal("10500.50"))

        collector = get_metrics_collector()
        assert "gpt_trader_equity_dollars" in collector.gauges
        assert collector.gauges["gpt_trader_equity_dollars"] == 10500.50

    def test_update_ws_health_records_gap_gauge(self) -> None:
        """update_ws_health records the WS gap count gauge metric."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_ws_health(
            {
                "connected": True,
                "last_message_ts": time.time(),
                "last_heartbeat_ts": time.time(),
                "gap_count": 5,
                "reconnect_count": 2,
            }
        )

        collector = get_metrics_collector()
        assert "gpt_trader_ws_gap_count" in collector.gauges
        assert collector.gauges["gpt_trader_ws_gap_count"] == 5.0

    def test_equity_gauge_updates_on_change(self) -> None:
        """equity gauge updates when equity changes."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_equity(Decimal("10000.00"))
        reporter.update_equity(Decimal("10500.00"))

        collector = get_metrics_collector()
        assert collector.gauges["gpt_trader_equity_dollars"] == 10500.00
