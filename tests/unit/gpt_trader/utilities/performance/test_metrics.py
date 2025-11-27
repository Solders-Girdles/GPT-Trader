"""Tests for performance metrics module."""

from __future__ import annotations

from gpt_trader.utilities.performance.metrics import (
    PerformanceCollector,
    PerformanceMetric,
    PerformanceStats,
    get_collector,
)


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_init_required_fields(self) -> None:
        metric = PerformanceMetric(name="test_op", value=0.5, unit="s")
        assert metric.name == "test_op"
        assert metric.value == 0.5
        assert metric.unit == "s"
        assert metric.timestamp > 0
        assert metric.tags == {}

    def test_init_with_tags(self) -> None:
        metric = PerformanceMetric(name="test_op", value=1.0, unit="ms", tags={"env": "test"})
        assert metric.tags == {"env": "test"}

    def test_str_without_tags(self) -> None:
        metric = PerformanceMetric(name="op", value=1.234, unit="s")
        result = str(metric)
        assert result == "op: 1.234s"

    def test_str_with_tags(self) -> None:
        metric = PerformanceMetric(name="op", value=1.234, unit="s", tags={"env": "prod"})
        result = str(metric)
        assert "op" in result
        assert "1.234s" in result
        assert "env=prod" in result

    def test_to_dict(self) -> None:
        metric = PerformanceMetric(name="test", value=0.5, unit="s", tags={"key": "val"})
        result = metric.to_dict()
        assert result["name"] == "test"
        assert result["value"] == 0.5
        assert result["unit"] == "s"
        assert "timestamp" in result
        assert result["tags"] == {"key": "val"}


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""

    def test_default_values(self) -> None:
        stats = PerformanceStats()
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.min == float("inf")
        assert stats.max == float("-inf")
        assert stats.avg == 0.0
        assert stats.recent_avg == 0.0

    def test_update_single(self) -> None:
        stats = PerformanceStats()
        stats.update(10.0)
        assert stats.count == 1
        assert stats.total == 10.0
        assert stats.min == 10.0
        assert stats.max == 10.0
        assert stats.avg == 10.0
        assert stats.value == 10.0

    def test_update_multiple(self) -> None:
        stats = PerformanceStats()
        stats.update(10.0)
        stats.update(20.0)
        assert stats.count == 2
        assert stats.total == 30.0
        assert stats.min == 10.0
        assert stats.max == 20.0
        assert stats.avg == 15.0

    def test_str_format(self) -> None:
        stats = PerformanceStats()
        stats.update(10.0)
        result = str(stats)
        assert "count=1" in result
        assert "avg=10.000" in result
        assert "min=10.000" in result
        assert "max=10.000" in result


class TestPerformanceCollector:
    """Tests for PerformanceCollector class."""

    def test_init(self) -> None:
        collector = PerformanceCollector()
        assert collector.max_history == 1000

    def test_init_custom_history(self) -> None:
        collector = PerformanceCollector(max_history=100)
        assert collector.max_history == 100

    def test_record_stores_metric(self) -> None:
        collector = PerformanceCollector()
        metric = PerformanceMetric(name="test", value=1.0, unit="s")
        collector.record(metric)
        stats = collector.get_stats("test")
        assert stats.count == 1
        assert stats.value == 1.0

    def test_record_updates_stats(self) -> None:
        collector = PerformanceCollector()
        collector.record(PerformanceMetric(name="op", value=10.0, unit="ms"))
        collector.record(PerformanceMetric(name="op", value=20.0, unit="ms"))
        stats = collector.get_stats("op")
        assert stats.count == 2
        assert stats.avg == 15.0

    def test_record_calculates_recent_avg(self) -> None:
        collector = PerformanceCollector()
        collector.record(PerformanceMetric(name="op", value=10.0, unit="ms"))
        collector.record(PerformanceMetric(name="op", value=30.0, unit="ms"))
        stats = collector.get_stats("op")
        assert stats.recent_avg == 20.0

    def test_get_stats_unknown_name(self) -> None:
        collector = PerformanceCollector()
        stats = collector.get_stats("unknown")
        assert stats.count == 0

    def test_get_recent_metrics(self) -> None:
        collector = PerformanceCollector()
        for i in range(5):
            collector.record(PerformanceMetric(name="op", value=float(i), unit="s"))
        recent = collector.get_recent_metrics("op", count=3)
        assert len(recent) == 3
        assert recent[-1].value == 4.0

    def test_get_recent_metrics_unknown(self) -> None:
        collector = PerformanceCollector()
        recent = collector.get_recent_metrics("unknown")
        assert recent == []

    def test_clear_specific_name(self) -> None:
        collector = PerformanceCollector()
        collector.record(PerformanceMetric(name="op1", value=1.0, unit="s"))
        collector.record(PerformanceMetric(name="op2", value=2.0, unit="s"))
        collector.clear("op1")
        assert collector.get_stats("op1").count == 0
        assert collector.get_stats("op2").count == 1

    def test_clear_all(self) -> None:
        collector = PerformanceCollector()
        collector.record(PerformanceMetric(name="op1", value=1.0, unit="s"))
        collector.record(PerformanceMetric(name="op2", value=2.0, unit="s"))
        collector.clear()
        assert collector.get_stats("op1").count == 0
        assert collector.get_stats("op2").count == 0

    def test_get_summary(self) -> None:
        collector = PerformanceCollector()
        collector.record(PerformanceMetric(name="op1", value=10.0, unit="s"))
        collector.record(PerformanceMetric(name="op1", value=20.0, unit="s"))
        summary = collector.get_summary()
        assert "op1" in summary
        assert summary["op1"]["count"] == 2
        assert summary["op1"]["avg"] == 15.0

    def test_max_history_enforced(self) -> None:
        collector = PerformanceCollector(max_history=5)
        for i in range(10):
            collector.record(PerformanceMetric(name="op", value=float(i), unit="s"))
        recent = collector.get_recent_metrics("op", count=10)
        assert len(recent) == 5


class TestGetCollector:
    """Tests for get_collector function."""

    def test_returns_collector(self) -> None:
        collector = get_collector()
        assert isinstance(collector, PerformanceCollector)

    def test_returns_same_instance(self) -> None:
        collector1 = get_collector()
        collector2 = get_collector()
        assert collector1 is collector2
