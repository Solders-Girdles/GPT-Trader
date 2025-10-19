"""Tests for core performance monitoring utilities."""

from __future__ import annotations

import time

import pytest

from bot_v2.utilities.performance_monitoring import (
    PerformanceCollector,
    PerformanceMetric,
    PerformanceStats,
    PerformanceTimer,
    measure_performance,
    measure_performance_decorator,
)


class TestPerformanceMetric:
    """Test PerformanceMetric functionality."""

    def test_performance_metric_creation(self) -> None:
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            name="test_operation", value=0.5, unit="s", tags={"component": "test"}
        )

        assert metric.name == "test_operation"
        assert metric.value == 0.5
        assert metric.unit == "s"
        assert metric.tags["component"] == "test"
        assert metric.timestamp > 0

    def test_performance_metric_string_representation(self) -> None:
        """Test string representation of performance metric."""
        metric = PerformanceMetric(
            name="test_operation", value=0.5, unit="s", tags={"component": "test"}
        )

        str_repr = str(metric)
        assert "test_operation" in str_repr
        assert "0.500s" in str_repr
        assert "component=test" in str_repr

    def test_performance_metric_to_dict(self) -> None:
        """Test converting metric to dictionary."""
        metric = PerformanceMetric(
            name="test_operation", value=0.5, unit="s", tags={"component": "test"}
        )

        metric_dict = metric.to_dict()

        assert metric_dict["name"] == "test_operation"
        assert metric_dict["value"] == 0.5
        assert metric_dict["unit"] == "s"
        assert metric_dict["tags"]["component"] == "test"
        assert "timestamp" in metric_dict


class TestPerformanceStats:
    """Test PerformanceStats functionality."""

    def test_performance_stats_initialization(self) -> None:
        """Test initial performance statistics."""
        stats = PerformanceStats()

        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.min == float("inf")
        assert stats.max == float("-inf")
        assert stats.avg == 0.0

    def test_performance_stats_update(self) -> None:
        """Test updating performance statistics."""
        stats = PerformanceStats()

        # Update with some values
        stats.update(0.1)
        stats.update(0.3)
        stats.update(0.2)

        assert stats.count == 3
        assert stats.total == 0.6
        assert stats.min == 0.1
        assert stats.max == 0.3
        assert stats.avg == 0.2

    def test_performance_stats_string_representation(self) -> None:
        """Test string representation of performance statistics."""
        stats = PerformanceStats()
        stats.update(0.1)
        stats.update(0.3)

        str_repr = str(stats)
        assert "count=2" in str_repr
        assert "avg=0.200" in str_repr
        assert "min=0.100" in str_repr
        assert "max=0.300" in str_repr


class TestPerformanceCollector:
    """Test PerformanceCollector functionality."""

    def test_performance_collector_initialization(self) -> None:
        """Test collector initialization."""
        collector = PerformanceCollector(max_history=100)

        assert collector.max_history == 100
        assert len(collector._metrics) == 0
        assert len(collector._stats) == 0

    def test_performance_collector_record_metric(self) -> None:
        """Test recording a performance metric."""
        collector = PerformanceCollector()

        metric = PerformanceMetric(name="test_operation", value=0.5, unit="s")

        collector.record(metric)

        # Check that metric was recorded
        assert "test_operation" in collector._metrics
        assert len(collector._metrics["test_operation"]) == 1

        # Check that statistics were updated
        assert "test_operation" in collector._stats
        stats = collector._stats["test_operation"]
        assert stats.count == 1
        assert stats.avg == 0.5

    def test_performance_collector_get_stats(self) -> None:
        """Test getting statistics for a metric."""
        collector = PerformanceCollector()

        # Record some metrics
        for i in range(3):
            metric = PerformanceMetric(name="test_operation", value=0.1 * i, unit="s")
            collector.record(metric)

        stats = collector.get_stats("test_operation")
        assert stats.count == 3
        assert stats.avg == 0.1  # (0.0 + 0.1 + 0.2) / 3

    def test_performance_collector_get_recent_metrics(self) -> None:
        """Test getting recent metrics."""
        collector = PerformanceCollector()

        # Record multiple metrics
        for i in range(5):
            metric = PerformanceMetric(name="test_operation", value=0.1 * i, unit="s")
            collector.record(metric)

        recent = collector.get_recent_metrics("test_operation", count=3)
        assert len(recent) == 3
        assert recent[-1].value == 0.4  # Last metric

    def test_performance_collector_clear(self) -> None:
        """Test clearing metrics."""
        collector = PerformanceCollector()

        # Record a metric
        metric = PerformanceMetric(name="test_operation", value=0.5, unit="s")
        collector.record(metric)

        # Clear specific metric
        collector.clear("test_operation")
        assert len(collector._metrics) == 0
        assert len(collector._stats) == 0

        # Record again and clear all
        collector.record(metric)
        collector.clear()
        assert len(collector._metrics) == 0
        assert len(collector._stats) == 0

    def test_performance_collector_summary(self) -> None:
        """Test getting performance summary."""
        collector = PerformanceCollector()

        # Record some metrics
        for i in range(3):
            metric = PerformanceMetric(name=f"operation_{i}", value=0.1 * i, unit="s")
            collector.record(metric)

        summary = collector.get_summary()
        assert len(summary) == 3
        assert "operation_0" in summary
        assert "operation_1" in summary
        assert "operation_2" in summary

        # Check summary structure
        for name, stats in summary.items():
            assert "count" in stats
            assert "avg" in stats
            assert "min" in stats
            assert "max" in stats


class TestMeasurePerformance:
    """Test performance measurement utilities."""

    def test_measure_performance_context_manager(self) -> None:
        """Test measure_performance context manager."""
        collector = PerformanceCollector()

        with measure_performance("test_operation", collector=collector):
            time.sleep(0.01)  # Small delay

        # Check that metric was recorded
        stats = collector.get_stats("test_operation")
        assert stats.count == 1
        assert stats.value > 0.01  # Should be at least the sleep time

    def test_measure_performance_decorator(self) -> None:
        """Test measure_performance_decorator."""
        collector = PerformanceCollector()

        @measure_performance_decorator("decorated_func", collector=collector)
        def test_function(x: int) -> int:
            time.sleep(0.01)
            return x * 2

        result = test_function(5)
        assert result == 10

        # Check that metric was recorded
        stats = collector.get_stats("decorated_func")
        assert stats.count == 1
        assert stats.value > 0.01

    def test_performance_timer(self) -> None:
        """Test PerformanceTimer."""
        collector = PerformanceCollector()

        timer = PerformanceTimer("timer_operation", collector=collector)

        timer.start()
        time.sleep(0.01)
        duration = timer.stop()

        assert duration > 0.01

        # Check that metric was recorded
        stats = collector.get_stats("timer_operation")
        assert stats.count == 1

    def test_performance_timer_context_manager(self) -> None:
        """Test PerformanceTimer as context manager."""
        collector = PerformanceCollector()

        with PerformanceTimer("context_timer", collector=collector):
            time.sleep(0.01)

        # Check that metric was recorded
        stats = collector.get_stats("context_timer")
        assert stats.count == 1
        assert stats.value > 0.01


class TestPerformanceEdgeCases:
    """Test edge cases and error conditions."""

    def test_performance_timer_not_started(self) -> None:
        """Test performance timer not started."""
        timer = PerformanceTimer("test")

        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()

    def test_performance_collector_max_history(self) -> None:
        """Test collector max history limit."""
        collector = PerformanceCollector(max_history=3)

        # Add more metrics than max_history
        for i in range(5):
            metric = PerformanceMetric(name="test", value=0.1 * i, unit="s")
            collector.record(metric)

        # Should only keep max_history metrics
        recent = collector.get_recent_metrics("test", count=10)
        assert len(recent) == 3

    def test_empty_performance_report(self) -> None:
        """Test generating report with no data."""
        from bot_v2.utilities.performance_monitoring import PerformanceReporter

        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)

        report = reporter.generate_report()
        assert "No metrics recorded" in report


class TestPerformanceIntegration:
    """Test integration between performance components."""

    def test_multiple_collectors_independence(self) -> None:
        """Test that multiple collectors are independent."""
        collector1 = PerformanceCollector()
        collector2 = PerformanceCollector()

        # Record metric in first collector
        metric = PerformanceMetric(name="test", value=0.5, unit="s")
        collector1.record(metric)

        # First should have data, second should be empty
        assert len(collector1.get_summary()) > 0
        assert len(collector2.get_summary()) == 0
