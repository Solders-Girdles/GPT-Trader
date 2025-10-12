"""Facade around performance monitoring helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any

from .metrics import PerformanceCollector, PerformanceMetric, get_collector
from .profiler import PerformanceProfiler, get_profiler, profile_performance
from .resource import ResourceMonitor, get_resource_monitor
from .timing import measure_performance


@dataclass
class PerformanceMonitor:
    """High-level facade that ties together performance helpers."""

    collector: PerformanceCollector = None  # type: ignore[assignment]
    profiler: PerformanceProfiler = None  # type: ignore[assignment]
    resource_monitor: ResourceMonitor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.collector is None:
            self.collector = get_collector()
        if self.profiler is None:
            self.profiler = get_profiler()
        if self.resource_monitor is None:
            self.resource_monitor = get_resource_monitor()

    # ------------------------------------------------------------------
    def record_duration(
        self,
        operation_name: str,
        duration_seconds: float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Record an explicit duration measurement."""

        metric = PerformanceMetric(
            name=operation_name,
            value=duration_seconds,
            unit="s",
            tags=dict(tags or {}),
        )
        self.collector.record(metric)

    def time(
        self, operation_name: str, *, tags: Mapping[str, str] | None = None
    ) -> ContextDecorator:
        """Return a context manager for timing an operation."""

        return measure_performance(operation_name, dict(tags or {}), self.collector)

    def decorator(
        self,
        operation_name: str | None = None,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that times the wrapped function."""

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            name = operation_name or f"{func.__module__}.{func.__name__}"

            def timed(*args: Any, **kwargs: Any) -> Any:
                with self.time(name, tags=tags):
                    return func(*args, **kwargs)

            return timed

        return wrapper

    def profile(
        self,
        *,
        sample_rate: float = 0.1,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that profiles calls using the shared profiler."""

        return profile_performance(sample_rate=sample_rate, profiler=self.profiler)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, dict[str, float]]:
        """Return aggregated metric summary."""

        return self.collector.get_summary()

    def recent_metrics(self, name: str, *, count: int = 10) -> list[PerformanceMetric]:
        """Return recent metrics for ``name``."""

        return self.collector.get_recent_metrics(name, count=count)

    def system_snapshot(self) -> dict[str, Any]:
        """Return system resource information if available."""

        if not self.resource_monitor.is_available():
            return {}
        snapshot = self.resource_monitor.get_system_info()
        snapshot.update(self.resource_monitor.get_memory_usage())
        snapshot.update(self.resource_monitor.get_cpu_usage())
        return snapshot


__all__ = ["PerformanceMonitor"]
