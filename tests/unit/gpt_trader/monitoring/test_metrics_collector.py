"""Tests for metrics_collector module."""

from __future__ import annotations

import pytest

from gpt_trader.monitoring.metrics_collector import (
    TRADE_BLOCKED_COUNTER,
    TRADE_EXECUTED_COUNTER,
    format_metric_key,
    get_metrics_collector,
    record_counter,
    record_gauge,
    record_histogram,
    record_trade_blocked,
    record_trade_executed,
    reset_all,
)
from gpt_trader.monitoring.metrics_exporter import format_prometheus


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before and after each test."""
    reset_all()
    yield
    reset_all()


class TestFormatMetricKey:
    """Tests for format_metric_key helper."""

    def test_no_labels_returns_name_only(self):
        """Test that no labels returns just the metric name."""
        assert format_metric_key("my_counter", None) == "my_counter"
        assert format_metric_key("my_counter", {}) == "my_counter"

    def test_single_label(self):
        """Test formatting with a single label."""
        result = format_metric_key("my_counter", {"result": "success"})
        assert result == "my_counter{result=success}"

    def test_multiple_labels_sorted(self):
        """Test that multiple labels are sorted alphabetically."""
        result = format_metric_key(
            "my_counter", {"side": "buy", "result": "success", "reason": "none"}
        )
        assert result == "my_counter{reason=none,result=success,side=buy}"

    def test_labels_stringified(self):
        """Test that label values are stringified."""
        result = format_metric_key("my_counter", {"count": 42, "enabled": True})
        assert result == "my_counter{count=42,enabled=True}"


class TestCounters:
    """Tests for counter functionality."""

    def test_record_counter_increments(self):
        """Test basic counter increment."""
        record_counter("test_counter")
        record_counter("test_counter")
        record_counter("test_counter", increment=3)

        collector = get_metrics_collector()
        assert collector.counters["test_counter"] == 5

    def test_record_counter_with_labels(self):
        """Test counter with labels."""
        record_counter(
            "gpt_trader_order_submission_total",
            labels={"result": "success", "side": "buy"},
        )
        record_counter(
            "gpt_trader_order_submission_total",
            labels={"result": "failed", "side": "buy"},
        )
        record_counter(
            "gpt_trader_order_submission_total",
            labels={"result": "success", "side": "buy"},
        )

        collector = get_metrics_collector()
        assert collector.counters["gpt_trader_order_submission_total{result=success,side=buy}"] == 2
        assert collector.counters["gpt_trader_order_submission_total{result=failed,side=buy}"] == 1

    def test_counter_backward_compatible(self):
        """Test that counter API is backward compatible (no labels)."""
        record_counter("old_style_counter")
        record_counter("old_style_counter", increment=5)

        collector = get_metrics_collector()
        # Should work without labels (dict key is just the name)
        assert collector.counters["old_style_counter"] == 6

    def test_trade_counters_exported_via_prometheus_output(self):
        """Test trade counters appear in Prometheus output."""
        record_trade_executed()
        record_trade_executed()
        record_trade_blocked()

        output = format_prometheus(get_metrics_collector().get_metrics_summary())
        output_lines = output.splitlines()

        assert "# TYPE gpt_trader_trades_executed_total counter" in output_lines
        assert "# TYPE gpt_trader_trades_blocked_total counter" in output_lines
        assert "gpt_trader_trades_executed_total 2" in output_lines
        assert "gpt_trader_trades_blocked_total 1" in output_lines


class TestTradeCounters:
    """Tests for trade-specific counter naming and increments."""

    def test_trade_counter_names_are_stable(self):
        """Test that trade counter keys remain stable."""
        assert TRADE_EXECUTED_COUNTER == "gpt_trader_trades_executed_total"
        assert TRADE_BLOCKED_COUNTER == "gpt_trader_trades_blocked_total"

    def test_record_trade_counters_increment(self):
        """Test trade counter helpers increment exported counter metrics."""
        record_trade_executed()
        record_trade_blocked()

        output = format_prometheus(get_metrics_collector().get_metrics_summary())
        output_lines = output.splitlines()

        assert "gpt_trader_trades_executed_total 1" in output_lines
        assert "gpt_trader_trades_blocked_total 1" in output_lines


class TestGauges:
    """Tests for gauge functionality."""

    def test_record_gauge_sets_value(self):
        """Test basic gauge setting."""
        record_gauge("test_gauge", 42.5)

        collector = get_metrics_collector()
        assert collector.gauges["test_gauge"] == 42.5

    def test_record_gauge_overwrites(self):
        """Test that gauge overwrites previous value."""
        record_gauge("test_gauge", 10.0)
        record_gauge("test_gauge", 20.0)

        collector = get_metrics_collector()
        assert collector.gauges["test_gauge"] == 20.0

    def test_record_gauge_with_labels(self):
        """Test gauge with labels."""
        record_gauge("gpt_trader_equity_dollars", 10000.0)
        record_gauge("gpt_trader_ws_gap_count", 5.0)

        collector = get_metrics_collector()
        assert collector.gauges["gpt_trader_equity_dollars"] == 10000.0
        assert collector.gauges["gpt_trader_ws_gap_count"] == 5.0


class TestResetAll:
    """Tests for reset_all functionality."""

    def test_reset_clears_all(self):
        """Test that reset_all clears all metric types."""
        record_counter("counter1")
        record_gauge("gauge1", 100.0)
        record_histogram("hist1", 0.5)

        collector = get_metrics_collector()
        assert len(collector.counters) > 0
        assert len(collector.gauges) > 0
        assert len(collector.histograms) > 0

        reset_all()

        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
        assert len(collector.histograms) == 0
