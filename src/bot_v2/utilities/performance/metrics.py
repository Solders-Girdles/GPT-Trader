"""Metric primitives and collectors for performance monitoring."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger("performance", component="monitoring")


@dataclass
class PerformanceMetric:
    """Single performance metric data point."""

    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        tags_str = ",".join(f"{k}={v}" for k, v in self.tags.items())
        tags_part = f"[{tags_str}]" if tags_str else ""
        return f"{self.name}{tags_part}: {self.value:.3f}{self.unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "tags": dict(self.tags),
        }


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0.0
    recent_avg: float = 0.0
    value: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total = round(self.total + value, 10)
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = round(self.total / self.count, 10)
        self.value = value

    def __str__(self) -> str:
        return (
            f"count={self.count}, "
            f"avg={self.avg:.3f}, "
            f"min={self.min:.3f}, "
            f"max={self.max:.3f}, "
            f"total={self.total:.3f}"
        )


class PerformanceCollector:
    """Collects and manages performance metrics."""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._metrics: defaultdict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._stats: defaultdict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._lock = threading.RLock()

    def record(self, metric: PerformanceMetric) -> None:
        with self._lock:
            history = self._metrics[metric.name]
            history.append(metric)

            stats = self._stats[metric.name]
            stats.update(metric.value)

            if history:
                running_total = sum(item.value for item in history)
                stats.recent_avg = round(running_total / len(history), 10)

    def get_stats(self, name: str) -> PerformanceStats:
        with self._lock:
            return self._stats[name]

    def get_recent_metrics(self, name: str, count: int = 10) -> list[PerformanceMetric]:
        with self._lock:
            history = list(self._metrics.get(name, []))
            return history[-count:]

    def clear(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._metrics.clear()
                self._stats.clear()
            else:
                self._metrics.pop(name, None)
                self._stats.pop(name, None)

    def get_summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            summary: dict[str, dict[str, float]] = {}
            for name, stats in self._stats.items():
                summary[name] = {
                    "count": stats.count,
                    "avg": stats.avg,
                    "min": stats.min,
                    "max": stats.max,
                    "total": stats.total,
                }
            return summary


_collector: PerformanceCollector | None = None


def get_collector() -> PerformanceCollector:
    """Return the global performance collector."""

    global _collector
    if _collector is None:
        _collector = PerformanceCollector()
    return _collector


__all__ = [
    "PerformanceMetric",
    "PerformanceStats",
    "PerformanceCollector",
    "get_collector",
]
