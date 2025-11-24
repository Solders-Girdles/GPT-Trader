"""Timing helpers and measurement context managers."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from .metrics import PerformanceCollector, PerformanceMetric, get_collector

T = TypeVar("T")


@contextmanager
def measure_performance(
    operation_name: str,
    tags: dict[str, str] | None = None,
    collector: PerformanceCollector | None = None,
) -> Any:
    """Context manager for measuring operation performance."""

    start_time = time.time()
    tags = tags or {}
    collector = collector or get_collector()

    try:
        yield
    finally:
        duration = time.time() - start_time
        metric = PerformanceMetric(name=operation_name, value=duration, unit="s", tags=tags)
        collector.record(metric)


def measure_performance_decorator(
    operation_name: str | None = None,
    tags: dict[str, str] | None = None,
    collector: PerformanceCollector | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for measuring function performance."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args: Any, **kwargs: Any) -> T:
            with measure_performance(name, tags, collector):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class PerformanceTimer:
    """Manual performance timer for custom measurements."""

    def __init__(
        self,
        operation_name: str,
        tags: dict[str, str] | None = None,
        collector: PerformanceCollector | None = None,
    ) -> None:
        self.operation_name = operation_name
        self.tags = tags or {}
        self.collector = collector or get_collector()
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        metric = PerformanceMetric(
            name=self.operation_name,
            value=duration,
            unit="s",
            tags=self.tags,
        )
        self.collector.record(metric)
        return duration

    def __enter__(self) -> PerformanceTimer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


__all__ = ["measure_performance", "measure_performance_decorator", "PerformanceTimer"]
