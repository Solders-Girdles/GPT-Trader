"""Tests for performance monitoring utilities."""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from bot_v2.utilities.performance_monitoring import (
    PerformanceMetric,
    PerformanceStats,
    PerformanceCollector,
    measure_performance,
    measure_performance_decorator,
    PerformanceTimer,
    ResourceMonitor,
    PerformanceProfiler,
    profile_performance,
    PerformanceReporter,
    monitor_trading_operation,
    monitor_database_operation,
    monitor_api_operation,
    get_performance_health_check,
    get_collector,
    get_resource_monitor,
    get_profiler,
)


class TestPerformanceMetric:
    """Test PerformanceMetric functionality."""
    
    def test_performance_metric_creation(self) -> None:
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            name="test_operation",
            value=0.5,
            unit="s",
            tags={"component": "test"}
        )
        
        assert metric.name == "test_operation"
        assert metric.value == 0.5
        assert metric.unit == "s"
        assert metric.tags["component"] == "test"
        assert metric.timestamp > 0
        
    def test_performance_metric_string_representation(self) -> None:
        """Test string representation of performance metric."""
        metric = PerformanceMetric(
            name="test_operation",
            value=0.5,
            unit="s",
            tags={"component": "test"}
        )
        
        str_repr = str(metric)
        assert "test_operation" in str_repr
        assert "0.500s" in str_repr
        assert "component=test" in str_repr
        
    def test_performance_metric_to_dict(self) -> None:
        """Test converting metric to dictionary."""
        metric = PerformanceMetric(
            name="test_operation",
            value=0.5,
            unit="s",
            tags={"component": "test"}
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
        
        metric = PerformanceMetric(
            name="test_operation",
            value=0.5,
            unit="s"
        )
        
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
            metric = PerformanceMetric(
                name="test_operation",
                value=0.1 * i,
                unit="s"
            )
            collector.record(metric)
            
        stats = collector.get_stats("test_operation")
        assert stats.count == 3
        assert stats.avg == 0.1  # (0.0 + 0.1 + 0.2) / 3
        
    def test_performance_collector_get_recent_metrics(self) -> None:
        """Test getting recent metrics."""
        collector = PerformanceCollector()
        
        # Record multiple metrics
        for i in range(5):
            metric = PerformanceMetric(
                name="test_operation",
                value=0.1 * i,
                unit="s"
            )
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
            metric = PerformanceMetric(
                name=f"operation_{i}",
                value=0.1 * i,
                unit="s"
            )
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


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def test_resource_monitor_initialization(self) -> None:
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        
        # Should try to import psutil
        assert monitor._psutil is not None or monitor._psutil is None
        
    def test_resource_monitor_availability(self) -> None:
        """Test checking resource monitor availability."""
        monitor = ResourceMonitor()
        
        # Should not raise exception
        is_available = monitor.is_available()
        assert isinstance(is_available, bool)
        
    def test_resource_monitor_memory_usage(self) -> None:
        """Test getting memory usage."""
        monitor = ResourceMonitor()
        
        memory_usage = monitor.get_memory_usage()
        
        if monitor.is_available():
            assert isinstance(memory_usage, dict)
            assert "rss_mb" in memory_usage
            assert "vms_mb" in memory_usage
            assert "percent" in memory_usage
            assert memory_usage["rss_mb"] > 0
        else:
            assert memory_usage == {}
            
    def test_resource_monitor_cpu_usage(self) -> None:
        """Test getting CPU usage."""
        monitor = ResourceMonitor()
        
        cpu_usage = monitor.get_cpu_usage()
        
        if monitor.is_available():
            assert isinstance(cpu_usage, dict)
            assert "cpu_percent" in cpu_usage
            assert "cpu_count" in cpu_usage
        else:
            assert cpu_usage == {}
            
    def test_resource_monitor_system_info(self) -> None:
        """Test getting system information."""
        monitor = ResourceMonitor()
        
        system_info = monitor.get_system_info()
        
        if monitor.is_available():
            assert isinstance(system_info, dict)
            assert "cpu_count" in system_info
            assert "memory_total_gb" in system_info
            assert "memory_available_gb" in system_info
            assert "memory_percent" in system_info
        else:
            assert system_info == {}


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    def test_performance_profiler_initialization(self) -> None:
        """Test profiler initialization."""
        profiler = PerformanceProfiler(sample_rate=0.5)
        
        assert profiler.sample_rate == 0.5
        assert len(profiler._call_counts) == 0
        assert len(profiler._total_times) == 0
        
    def test_performance_profiler_sampling(self) -> None:
        """Test profiler sampling logic."""
        profiler = PerformanceProfiler(sample_rate=1.0)  # Always sample
        
        # Should always sample
        assert profiler.should_sample()
        
        profiler = PerformanceProfiler(sample_rate=0.0)  # Never sample
        assert not profiler.should_sample()
        
    def test_performance_profiler_record_call(self) -> None:
        """Test recording function calls."""
        profiler = PerformanceProfiler()
        
        profiler.record_call("test_func", 0.1)
        profiler.record_call("test_func", 0.2)
        profiler.record_call("other_func", 0.15)
        
        assert profiler._call_counts["test_func"] == 2
        assert profiler._total_times["test_func"] == 0.3
        assert profiler._call_counts["other_func"] == 1
        assert profiler._total_times["other_func"] == 0.15
        
    def test_performance_profiler_get_profile_data(self) -> None:
        """Test getting profile data."""
        profiler = PerformanceProfiler(sample_rate=0.5)
        
        profiler.record_call("test_func", 0.1)
        profiler.record_call("test_func", 0.2)
        
        profile_data = profiler.get_profile_data()
        
        assert "test_func" in profile_data
        assert profile_data["test_func"]["call_count"] == 2
        assert profile_data["test_func"]["total_time"] == 0.3
        assert profile_data["test_func"]["avg_time"] == 0.15
        assert profile_data["test_func"]["sample_rate"] == 0.5
        
    def test_profile_performance_decorator(self) -> None:
        """Test profile_performance decorator."""
        profiler = PerformanceProfiler(sample_rate=1.0)  # Always sample
        
        @profile_performance(sample_rate=1.0, profiler=profiler)
        def test_function(x: int) -> int:
            time.sleep(0.01)
            return x * 2
            
        result = test_function(5)
        assert result == 10
        
        # Check that call was recorded
        profile_data = profiler.get_profile_data()
        assert len(profile_data) > 0


class TestPerformanceReporter:
    """Test PerformanceReporter functionality."""
    
    def test_performance_reporter_initialization(self) -> None:
        """Test reporter initialization."""
        collector = PerformanceCollector()
        resource_monitor = ResourceMonitor()
        profiler = PerformanceProfiler()
        
        reporter = PerformanceReporter(
            collector=collector,
            resource_monitor=resource_monitor,
            profiler=profiler
        )
        
        assert reporter.collector is collector
        assert reporter.resource_monitor is resource_monitor
        assert reporter.profiler is profiler
        
    def test_performance_reporter_generate_report(self) -> None:
        """Test generating performance report."""
        collector = PerformanceCollector()
        
        # Add some metrics
        metric = PerformanceMetric(name="test_operation", value=0.5, unit="s")
        collector.record(metric)
        
        reporter = PerformanceReporter(collector=collector)
        report = reporter.generate_report()
        
        assert "Performance Report" in report
        assert "test_operation" in report
        assert "Performance Metrics" in report
        
    def test_performance_reporter_log_report(self) -> None:
        """Test logging performance report."""
        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)
        
        # Should not raise exception
        reporter.log_report()
        
    def test_performance_reporter_save_report(self) -> None:
        """Test saving performance report to file."""
        import tempfile
        import os
        
        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
            
        try:
            reporter.save_report(temp_path)
            
            # Check that file was created and contains report
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "Performance Report" in content
        finally:
            os.unlink(temp_path)


class TestMonitoringDecorators:
    """Test monitoring decorators."""
    
    def test_monitor_trading_operation(self) -> None:
        """Test trading operation monitoring decorator."""
        collector = get_collector()
        
        @monitor_trading_operation("place_order")
        def place_order(symbol: str, quantity: float) -> str:
            time.sleep(0.01)
            return f"Order placed for {symbol}"
            
        result = place_order("BTC-PERP", 0.1)
        assert result == "Order placed for BTC-PERP"
        
        # Check that metric was recorded with correct name
        stats = collector.get_stats("trading.place_order")
        assert stats.count == 1
        
    def test_monitor_database_operation(self) -> None:
        """Test database operation monitoring decorator."""
        collector = get_collector()
        
        @monitor_database_operation("query")
        def execute_query(sql: str) -> list:
            time.sleep(0.01)
            return ["result1", "result2"]
            
        result = execute_query("SELECT * FROM table")
        assert result == ["result1", "result2"]
        
        # Check that metric was recorded
        stats = collector.get_stats("database.query")
        assert stats.count == 1
        
    def test_monitor_api_operation(self) -> None:
        """Test API operation monitoring decorator."""
        collector = get_collector()
        
        @monitor_api_operation("get_data")
        def api_get(endpoint: str) -> dict:
            time.sleep(0.01)
            return {"data": "test"}
            
        result = api_get("/api/test")
        assert result == {"data": "test"}
        
        # Check that metric was recorded
        stats = collector.get_stats("api.get_data")
        assert stats.count == 1


class TestPerformanceHealthCheck:
    """Test performance health check functionality."""
    
    def test_get_performance_health_check(self) -> None:
        """Test getting performance health check."""
        health = get_performance_health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "issues" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]


class TestGlobalInstances:
    """Test global instance getters."""
    
    def test_get_collector(self) -> None:
        """Test getting global collector."""
        collector = get_collector()
        assert isinstance(collector, PerformanceCollector)
        
    def test_get_resource_monitor(self) -> None:
        """Test getting global resource monitor."""
        monitor = get_resource_monitor()
        assert isinstance(monitor, ResourceMonitor)
        
    def test_get_profiler(self) -> None:
        """Test getting global profiler."""
        profiler = get_profiler()
        assert isinstance(profiler, PerformanceProfiler)


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
        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)
        
        report = reporter.generate_report()
        assert "No metrics recorded" in report


class TestPerformanceIntegration:
    """Test integration between performance components."""
    
    def test_end_to_end_performance_tracking(self) -> None:
        """Test end-to-end performance tracking."""
        collector = get_collector()
        
        # Clear any existing data
        collector.clear()
        
        # Use decorator to monitor function
        @monitor_trading_operation("test_trade")
        def test_trade():
            time.sleep(0.01)
            return "success"
            
        # Execute function multiple times
        for _ in range(3):
            test_trade()
            
        # Check statistics
        stats = collector.get_stats("trading.test_trade")
        assert stats.count == 3
        assert stats.avg > 0.01
        
        # Generate health check
        health = get_performance_health_check()
        assert isinstance(health, dict)
        
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
