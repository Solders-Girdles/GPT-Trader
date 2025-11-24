"""
Lightweight counter tracking for runtime guard telemetry.

The previous implementation modeled a full in-memory metrics time series store.
The trading runtime only consumed counter increments to confirm guard
executions, so we keep a tiny façade that preserves the public API used in
tests (`get_metrics_collector`, `record_counter`, `reset_all`,
`get_metrics_summary`) without the historical baggage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class _GuardMetricsCollector:
    counters: Counter[str] = field(default_factory=Counter)

    def record_counter(self, name: str, increment: int = 1) -> None:
        self.counters[name] += increment

    def reset_all(self) -> None:
        self.counters.clear()

    def get_metrics_summary(self) -> dict[str, object]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": dict(self.counters),
        }


_GLOBAL_COLLECTOR = _GuardMetricsCollector()


def get_metrics_collector() -> _GuardMetricsCollector:
    """Return the singleton guard metrics collector."""

    return _GLOBAL_COLLECTOR


def record_counter(name: str, increment: int = 1) -> None:
    """Increment the named counter."""

    _GLOBAL_COLLECTOR.record_counter(name, increment)


def reset_all() -> None:
    """Reset all tracked counters."""

    _GLOBAL_COLLECTOR.reset_all()
