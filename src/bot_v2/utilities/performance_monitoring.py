"""Backwards-compatible shim for performance monitoring utilities."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    psutil = None  # type: ignore

from .performance import (
    PerformanceCollector,
    PerformanceMetric,
    PerformanceProfiler,
    PerformanceReporter,
    PerformanceStats,
    PerformanceTimer,
    ResourceMonitor,
    get_collector,
    get_performance_health_check,
    get_profiler,
    get_resource_monitor,
    measure_performance,
    measure_performance_decorator,
    monitor_api_operation,
    monitor_database_operation,
    monitor_trading_operation,
    profile_performance,
)

__all__ = [
    "PerformanceMetric",
    "PerformanceStats",
    "PerformanceCollector",
    "PerformanceTimer",
    "PerformanceProfiler",
    "PerformanceReporter",
    "ResourceMonitor",
    "measure_performance",
    "measure_performance_decorator",
    "profile_performance",
    "monitor_trading_operation",
    "monitor_database_operation",
    "monitor_api_operation",
    "get_collector",
    "get_resource_monitor",
    "get_profiler",
    "get_performance_health_check",
    "psutil",
]
