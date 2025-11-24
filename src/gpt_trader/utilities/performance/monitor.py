"""Facade around performance monitoring helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from .metrics import PerformanceCollector, PerformanceMetric, get_collector
from .profiler import PerformanceProfiler, get_profiler, profile_performance
from .resource import ResourceMonitor, get_resource_monitor
from .timing import measure_performance

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class PerformanceMonitor:
    """High-level facade that ties together performance helpers."""

    collector: PerformanceCollector = field(default_factory=get_collector)
    profiler: PerformanceProfiler = field(default_factory=get_profiler)
    resource_monitor: ResourceMonitor = field(default_factory=get_resource_monitor)

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
    ) -> AbstractContextManager[None]:
        """Return a context manager for timing an operation."""

        manager = measure_performance(operation_name, dict(tags or {}), self.collector)
        return cast(AbstractContextManager[None], manager)

    def decorator(
        self,
        operation_name: str | None = None,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Return a decorator that times the wrapped function."""

        def wrapper(func: Callable[P, R]) -> Callable[P, R]:
            name = operation_name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def timed(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.time(name, tags=tags):
                    return func(*args, **kwargs)

            return timed

        return wrapper

    def profile(
        self,
        *,
        sample_rate: float = 0.1,
    ) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """Return a decorator that profiles calls using the shared profiler."""

        decorator = profile_performance(sample_rate=sample_rate, profiler=self.profiler)
        return cast(Callable[[Callable[..., R]], Callable[..., R]], decorator)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, dict[str, float]]:
        """Return aggregated metric summary."""

        summary = self.collector.get_summary()
        return cast(dict[str, dict[str, float]], summary)

    def recent_metrics(self, name: str, *, count: int = 10) -> list[PerformanceMetric]:
        """Return recent metrics for ``name``."""

        metrics = self.collector.get_recent_metrics(name, count=count)
        return cast(list[PerformanceMetric], metrics)

    def system_snapshot(self) -> dict[str, Any]:
        """Return system resource information if available."""

        if not self.resource_monitor.is_available():
            return {}
        snapshot = self.resource_monitor.get_system_info()
        snapshot.update(self.resource_monitor.get_memory_usage())
        snapshot.update(self.resource_monitor.get_cpu_usage())
        return cast(dict[str, Any], snapshot)


__all__ = ["PerformanceMonitor"]
