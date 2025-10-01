"""Tests for metrics collector"""

import pytest
import time
from bot_v2.monitoring.metrics_collector import MetricsCollector


class TestMetricsCollector:
    """Test suite for MetricsCollector"""

    def test_initialization(self):
        """Test collector initialization"""
        collector = MetricsCollector()

        assert collector.metrics == {}
        assert len(collector.counters) == 0
        assert collector.gauges == {}
        assert len(collector.histograms) == 0

    def test_record_counter(self):
        """Test recording a counter"""
        collector = MetricsCollector()

        collector.record_counter("requests")
        collector.record_counter("requests")
        collector.record_counter("requests", increment=5)

        assert collector.counters["requests"] == 7

    def test_record_gauge(self):
        """Test recording a gauge"""
        collector = MetricsCollector()

        collector.record_gauge("cpu_usage", 45.5)
        collector.record_gauge("cpu_usage", 55.2)

        assert collector.gauges["cpu_usage"] == 55.2

    def test_record_histogram(self):
        """Test recording a histogram"""
        collector = MetricsCollector()

        collector.record_histogram("response_time", 100.0)
        collector.record_histogram("response_time", 150.0)
        collector.record_histogram("response_time", 120.0)

        assert len(collector.histograms["response_time"]) == 3
        assert 100.0 in collector.histograms["response_time"]

    def test_collect_system_metrics(self):
        """Test system metrics collection"""
        collector = MetricsCollector()

        collector.collect_system_metrics()

        # Should record various system metrics
        assert "system.health.status" in collector.gauges
        assert "system.uptime_seconds" in collector.gauges
        assert "slices.available_count" in collector.gauges

    def test_start_stop_collection(self):
        """Test starting and stopping collection"""
        collector = MetricsCollector()

        collector.start_collection()
        assert collector._running is True

        time.sleep(0.1)  # Let it run briefly

        collector.stop_collection()
        assert collector._running is False

    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        collector = MetricsCollector()

        collector.record_counter("count_metric")
        collector.record_gauge("gauge_metric", 100.0)
        collector.record_histogram("hist_metric", 50.0)

        summary = collector.get_metrics_summary()

        assert "counters" in summary
        assert "gauges" in summary
        assert "histograms" in summary
        assert "timestamp" in summary

    def test_record_trading_metrics(self):
        """Test recording trading metrics"""
        collector = MetricsCollector()

        collector.record_trading_metrics(
            trades_executed=10,
            pnl=1500.50,
            portfolio_value=50000.0
        )

        assert collector.counters["trading.trades_executed"] == 10
        assert collector.gauges["trading.portfolio_value"] == 50000.0
        assert collector.gauges["trading.pnl"] == 1500.50

    def test_record_slice_performance(self):
        """Test recording slice performance"""
        collector = MetricsCollector()

        collector.record_slice_performance("live_trade", 125.5, True)

        assert collector.counters["slices.live_trade.executions"] == 1
        assert collector.counters["slices.live_trade.successes"] == 1
        assert "slices.live_trade.execution_time_ms" in collector.histograms

    def test_record_slice_performance_failure(self):
        """Test recording slice performance on failure"""
        collector = MetricsCollector()

        collector.record_slice_performance("backtest", 85.2, False)

        assert collector.counters["slices.backtest.failures"] == 1

    def test_export_metrics(self):
        """Test exporting metrics"""
        collector = MetricsCollector()

        collector.record_gauge("test_metric", 100.0)

        exported = collector.export_metrics(window_minutes=60)

        assert isinstance(exported, dict)