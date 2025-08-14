"""
Performance monitoring and profiling utilities for GPT-Trader.

This module provides comprehensive performance monitoring capabilities including:
- Memory usage tracking
- CPU utilization monitoring
- Execution time profiling
- Performance metrics collection
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    timestamp: datetime
    memory_usage_mb: float
    cpu_percent: float
    execution_time_seconds: float
    operation_name: str
    additional_metrics: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Centralized performance monitoring system."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.metrics: list[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._process = psutil.Process()

    def record_metric(
        self,
        operation_name: str,
        execution_time: float,
        additional_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record a performance metric."""
        if not self.enabled:
            return

        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            cpu_percent = self._process.cpu_percent()

            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent,
                execution_time_seconds=execution_time,
                operation_name=operation_name,
                additional_metrics=additional_metrics or {},
            )

            with self._lock:
                self.metrics.append(metric)

        except Exception as e:
            logger.warning(f"Failed to record performance metric: {e}")

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._process.memory_info().rss if self.enabled else 0

        try:
            yield
        finally:
            if self.enabled:
                end_time = time.time()
                end_memory = self._process.memory_info().rss
                execution_time = end_time - start_time
                memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB

                self.record_metric(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    additional_metrics={"memory_delta_mb": memory_delta},
                )

    def get_summary(
        self, operation_name: str | None = None, time_window: timedelta | None = None
    ) -> dict[str, Any]:
        """Get performance summary for specified criteria."""
        with self._lock:
            metrics = self.metrics.copy()

        # Filter by operation name if specified
        if operation_name:
            metrics = [m for m in metrics if m.operation_name == operation_name]

        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not metrics:
            return {"error": "No metrics found for specified criteria"}

        # Calculate summary statistics
        execution_times = [m.execution_time_seconds for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        cpu_usage = [m.cpu_percent for m in metrics]

        summary = {
            "operation_name": operation_name or "all",
            "time_window": str(time_window) if time_window else "all",
            "total_operations": len(metrics),
            "execution_time": {
                "mean": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "total": sum(execution_times),
            },
            "memory_usage_mb": {
                "mean": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage),
                "current": memory_usage[-1] if memory_usage else 0,
            },
            "cpu_percent": {
                "mean": sum(cpu_usage) / len(cpu_usage),
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "current": cpu_usage[-1] if cpu_usage else 0,
            },
        }

        return summary

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.metrics.clear()

    def export_metrics(self, filepath: str) -> None:
        """Export metrics to a file."""
        import json

        with self._lock:
            metrics_data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "operation_name": m.operation_name,
                    "memory_usage_mb": m.memory_usage_mb,
                    "cpu_percent": m.cpu_percent,
                    "execution_time_seconds": m.execution_time_seconds,
                    "additional_metrics": m.additional_metrics,
                }
                for m in self.metrics
            ]

        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Exported {len(metrics_data)} performance metrics to {filepath}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def profile_function(func):
    """Decorator to profile function execution."""

    def wrapper(*args, **kwargs):
        with performance_monitor.profile_operation(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def get_performance_summary(
    operation_name: str | None = None, time_window_minutes: int | None = None
) -> dict[str, Any]:
    """Convenience function to get performance summary."""
    time_window = timedelta(minutes=time_window_minutes) if time_window_minutes else None
    return performance_monitor.get_summary(operation_name, time_window)


def enable_performance_monitoring(enabled: bool = True) -> None:
    """Enable or disable performance monitoring."""
    performance_monitor.enabled = enabled
    logger.info(f"Performance monitoring {'enabled' if enabled else 'disabled'}")


def export_performance_metrics(filepath: str) -> None:
    """Export performance metrics to file."""
    performance_monitor.export_metrics(filepath)
