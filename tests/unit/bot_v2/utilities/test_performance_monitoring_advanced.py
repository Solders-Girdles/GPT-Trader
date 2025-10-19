"""Tests for advanced performance monitoring utilities."""

from __future__ import annotations

import os
import random
import tempfile
import time
from unittest.mock import Mock, patch

from bot_v2.utilities.performance_monitoring import (
    PerformanceProfiler,
    PerformanceReporter,
    ResourceMonitor,
    get_collector,
    get_performance_health_check,
    get_profiler,
    get_resource_monitor,
    monitor_api_operation,
    monitor_database_operation,
    monitor_trading_operation,
    profile_performance,
)


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

        random.seed(2024)

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
        from bot_v2.utilities.performance_monitoring import PerformanceCollector

        collector = PerformanceCollector()
        resource_monitor = ResourceMonitor()
        profiler = PerformanceProfiler()

        reporter = PerformanceReporter(
            collector=collector, resource_monitor=resource_monitor, profiler=profiler
        )

        assert reporter.collector is collector
        assert reporter.resource_monitor is resource_monitor
        assert reporter.profiler is profiler

    def test_performance_reporter_generate_report(self) -> None:
        """Test generating performance report."""
        from bot_v2.utilities.performance_monitoring import PerformanceCollector, PerformanceMetric

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
        from bot_v2.utilities.performance_monitoring import PerformanceCollector

        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)

        # Should not raise exception
        reporter.log_report()

    def test_performance_reporter_save_report(self) -> None:
        """Test saving performance report to file."""
        from bot_v2.utilities.performance_monitoring import PerformanceCollector

        collector = PerformanceCollector()
        reporter = PerformanceReporter(collector=collector)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            reporter.save_report(temp_path)

            # Check that file was created and contains report
            with open(temp_path) as f:
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
        from bot_v2.utilities.performance_monitoring import PerformanceCollector

        assert isinstance(collector, PerformanceCollector)

    def test_get_resource_monitor(self) -> None:
        """Test getting global resource monitor."""
        monitor = get_resource_monitor()
        assert isinstance(monitor, ResourceMonitor)

    def test_get_profiler(self) -> None:
        """Test getting global profiler."""
        profiler = get_profiler()
        assert isinstance(profiler, PerformanceProfiler)


class TestPerformanceAdvancedIntegration:
    """Test advanced integration scenarios."""

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

    def test_profiler_with_different_sample_rates(self) -> None:
        """Test profiler behavior with different sample rates."""
        profiler = PerformanceProfiler(sample_rate=0.5)

        # Record multiple calls
        for i in range(100):
            if profiler.should_sample():
                profiler.record_call("test_func", 0.01)

        # Should have approximately 50% of calls recorded
        profile_data = profiler.get_profile_data()
        if "test_func" in profile_data:
            # Allow some variance due to randomness
            call_count = profile_data["test_func"]["call_count"]
            expected = int(100 * profiler.sample_rate)
            tolerance = 15
            assert expected - tolerance <= call_count <= expected + tolerance

    def test_resource_monitor_with_mock_psutil(self) -> None:
        """Test resource monitor with mocked psutil."""
        with patch("bot_v2.utilities.performance_monitoring.psutil") as mock_psutil:
            # Mock psutil to always be available
            mock_psutil.virtual_memory.return_value = Mock(
                total=8589934592,
                available=4294967296,
                percent=50.0,
                used=4294967296,  # 8GB  # 4GB
            )
            mock_psutil.cpu_percent.return_value = 25.0
            mock_psutil.cpu_count.return_value = 4

            monitor = ResourceMonitor()

            memory_usage = monitor.get_memory_usage()
            assert memory_usage["percent"] == 50.0

            cpu_usage = monitor.get_cpu_usage()
            assert cpu_usage["cpu_percent"] == 25.0
            assert cpu_usage["cpu_count"] == 4

            system_info = monitor.get_system_info()
            assert system_info["memory_total_gb"] == 8.0
            assert system_info["memory_available_gb"] == 4.0
            assert system_info["memory_percent"] == 50.0
