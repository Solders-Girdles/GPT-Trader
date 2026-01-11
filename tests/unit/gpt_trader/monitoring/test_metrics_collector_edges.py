"""Edge-case tests for metrics collector helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.monitoring.metrics_collector import (
    format_metric_key,
    get_metrics_collector,
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
