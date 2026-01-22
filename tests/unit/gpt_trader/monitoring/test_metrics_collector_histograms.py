"""Tests for histogram metrics in metrics_collector module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.monitoring.metrics_collector import (
    DEFAULT_LATENCY_BUCKETS,
    HistogramData,
    format_metric_key,
    get_metrics_collector,
    record_counter,
    record_gauge,
    record_histogram,
    reset_all,
)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before and after each test."""
    reset_all()
    yield
    reset_all()


def test_format_metric_key_stringifies_non_string_labels() -> None:
    key = format_metric_key(
        "metric",
        {"value": Decimal("1.25"), "empty": None},
    )

    assert key == "metric{empty=None,value=1.25}"


def test_record_histogram_uses_configured_buckets() -> None:
    collector = get_metrics_collector()
    custom_buckets = (0.1, 0.5, 1.0)
    collector.configure_histogram_buckets("latency", custom_buckets)

    record_histogram("latency", 0.4)

    hist = collector.histograms["latency"]
    assert hist.buckets == custom_buckets


def test_record_histogram_with_labels_uses_configured_buckets() -> None:
    collector = get_metrics_collector()
    custom_buckets = (1.0, 2.0, 3.0)
    collector.configure_histogram_buckets("latency", custom_buckets)

    record_histogram("latency", 2.5, labels={"result": "ok"})

    hist = collector.histograms["latency{result=ok}"]
    assert hist.buckets == custom_buckets


def test_metrics_summary_includes_labeled_histograms_and_iso_timestamp() -> None:
    collector = get_metrics_collector()
    record_histogram("latency", 0.25, labels={"result": "ok"})

    summary = collector.get_metrics_summary()

    assert "latency{result=ok}" in summary["histograms"]
    timestamp = summary["timestamp"]
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


class TestMetricsSummary:
    """Tests for get_metrics_summary."""

    def test_summary_shape(self):
        """Test that summary has correct shape."""
        record_counter("counter1", labels={"type": "a"})
        record_gauge("gauge1", 100.0)
        record_histogram("hist1", 0.5, labels={"result": "ok"})

        collector = get_metrics_collector()
        summary = collector.get_metrics_summary()

        assert "timestamp" in summary
        assert "counters" in summary
        assert "gauges" in summary
        assert "histograms" in summary

        # Check types
        assert isinstance(summary["counters"], dict)
        assert isinstance(summary["gauges"], dict)
        assert isinstance(summary["histograms"], dict)

    def test_summary_includes_all_metrics(self):
        """Test that summary includes all recorded metrics."""
        record_counter("c1")
        record_counter("c2", labels={"x": "y"})
        record_gauge("g1", 1.0)
        record_histogram("h1", 0.1)

        collector = get_metrics_collector()
        summary = collector.get_metrics_summary()

        assert "c1" in summary["counters"]
        assert "c2{x=y}" in summary["counters"]
        assert "g1" in summary["gauges"]
        assert "h1" in summary["histograms"]


class TestHistograms:
    """Tests for histogram functionality."""

    def test_record_histogram_basic(self):
        """Test basic histogram recording."""
        record_histogram("test_histogram", 0.05)
        record_histogram("test_histogram", 0.15)
        record_histogram("test_histogram", 0.5)

        collector = get_metrics_collector()
        hist = collector.histograms["test_histogram"]
        assert hist.count == 3
        assert hist.total == pytest.approx(0.7)

    def test_record_histogram_with_labels(self):
        """Test histogram with labels."""
        record_histogram("gpt_trader_cycle_duration_seconds", 0.5, labels={"result": "ok"})
        record_histogram("gpt_trader_cycle_duration_seconds", 1.0, labels={"result": "ok"})
        record_histogram("gpt_trader_cycle_duration_seconds", 2.0, labels={"result": "error"})

        collector = get_metrics_collector()
        ok_hist = collector.histograms["gpt_trader_cycle_duration_seconds{result=ok}"]
        error_hist = collector.histograms["gpt_trader_cycle_duration_seconds{result=error}"]

        assert ok_hist.count == 2
        assert ok_hist.total == pytest.approx(1.5)
        assert error_hist.count == 1
        assert error_hist.total == pytest.approx(2.0)

    def test_histogram_buckets(self):
        """Test histogram bucket counting."""
        # Record values that fall into specific buckets
        record_histogram("latency", 0.001)  # <= 0.001 bucket
        record_histogram("latency", 0.008)  # <= 0.01 bucket
        record_histogram("latency", 0.5)  # <= 0.5 bucket
        record_histogram("latency", 100.0)  # > all buckets

        collector = get_metrics_collector()
        hist = collector.histograms["latency"]

        # Check bucket counts (cumulative)
        buckets = hist.to_dict()["buckets"]
        assert buckets["0.001"] == 1  # Only 0.001 fits
        assert buckets["0.01"] == 2  # 0.001 and 0.008 fit
        assert buckets["0.5"] == 3  # 0.001, 0.008, and 0.5 fit
        assert buckets["10.0"] == 3  # 100.0 doesn't fit in any bucket

    def test_histogram_custom_buckets(self):
        """Test histogram with custom buckets."""
        custom_buckets = (1.0, 5.0, 10.0, 50.0)
        record_histogram("custom", 3.0, buckets=custom_buckets)
        record_histogram("custom", 7.0)  # Uses previously set buckets
        record_histogram("custom", 100.0)

        collector = get_metrics_collector()
        hist = collector.histograms["custom"]

        assert hist.buckets == custom_buckets
        assert hist.count == 3
        buckets = hist.to_dict()["buckets"]
        assert buckets["5.0"] == 1  # 3.0 fits
        assert buckets["10.0"] == 2  # 3.0 and 7.0 fit


class TestHistogramData:
    """Tests for HistogramData class."""

    def test_to_dict_format(self):
        """Test histogram dict format."""
        hist = HistogramData(
            buckets=(0.1, 0.5, 1.0),
            bucket_counts=[0, 0, 0],
        )
        hist.record(0.05)
        hist.record(0.3)
        hist.record(0.8)

        result = hist.to_dict()
        assert result["count"] == 3
        assert result["sum"] == pytest.approx(1.15)
        assert result["mean"] == pytest.approx(1.15 / 3)
        assert "buckets" in result
        assert result["buckets"]["0.1"] == 1
        assert result["buckets"]["0.5"] == 2
        assert result["buckets"]["1.0"] == 3

    def test_mean_with_no_data(self):
        """Test mean is 0 when no data recorded."""
        hist = HistogramData(
            buckets=DEFAULT_LATENCY_BUCKETS,
            bucket_counts=[0] * len(DEFAULT_LATENCY_BUCKETS),
        )
        assert hist.to_dict()["mean"] == 0.0
