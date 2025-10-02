from __future__ import annotations

from collections import deque

import pytest

from bot_v2.monitoring.metrics_collector import MetricSeries, MetricsCollector


def test_metric_series_add_point_trims() -> None:
    series = MetricSeries(name="requests", points=deque(maxlen=2), max_points=2)
    series.add_point(1.0)
    series.add_point(2.0)
    series.add_point(3.0)

    assert len(series.points) == 2
    assert series.points[0].value == 2.0
    stats = series.get_stats()
    assert stats["count"] == 2
    assert stats["max"] == 3.0


def test_metrics_collector_records_and_summarises(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = MetricsCollector()

    collector.record_counter("counter.test", increment=2)
    collector.record_gauge("gauge.test", 5.5)
    collector.record_histogram("hist.test", 10.0)

    timer_id = collector.start_timer("timed.operation")
    # Simulate elapsed time without sleeping
    collector.timers[timer_id] -= 0.05

    def fake_record_histogram(name: str, value: float) -> None:
        collector.histograms[name].append(value)

    monkeypatch.setattr(collector, "record_histogram", fake_record_histogram)

    duration = collector.stop_timer(timer_id)
    assert duration > 0
    assert collector.histograms["timed.operation.duration_ms"]

    collector.record_trading_metrics(trades_executed=3, pnl=12.0, portfolio_value=1500.0)
    collector.record_slice_performance("analyze", execution_time_ms=25.0, success=False)

    summary = collector.get_metrics_summary()
    assert "counter.test" in summary["counters"]
    assert "timed.operation.duration_ms" in summary["histograms"]
