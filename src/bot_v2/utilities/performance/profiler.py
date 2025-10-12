"""Function profiling helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class PerformanceProfiler:
    """Profile function performance over time."""

    def __init__(self, sample_rate: float = 0.1) -> None:
        self.sample_rate = sample_rate
        self._call_counts: dict[str, int] = {}
        self._total_times: dict[str, float] = {}

    def should_sample(self) -> bool:
        return random.random() < self.sample_rate  # nosec B311

    def record_call(self, func_name: str, duration: float) -> None:
        self._call_counts[func_name] = self._call_counts.get(func_name, 0) + 1
        self._total_times[func_name] = round(self._total_times.get(func_name, 0.0) + duration, 9)

    def get_profile_data(self) -> dict[str, dict[str, float]]:
        profile_data: dict[str, dict[str, float]] = {}
        for func_name, count in self._call_counts.items():
            total_time = round(self._total_times.get(func_name, 0.0), 9)
            avg_time = round(total_time / count, 9) if count else 0.0
            profile_data[func_name] = {
                "call_count": count,
                "total_time": total_time,
                "avg_time": avg_time,
                "sample_rate": self.sample_rate,
            }
        return profile_data


_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def profile_performance(
    sample_rate: float = 0.1,
    profiler: PerformanceProfiler | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for profiling function performance."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        prof = profiler or get_profiler()

        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not prof.should_sample():
                return func(*args, **kwargs)
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                func_name = f"{func.__module__}.{func.__name__}"
                prof.record_call(func_name, duration)

        return wrapper

    return decorator


__all__ = ["PerformanceProfiler", "get_profiler", "profile_performance"]
