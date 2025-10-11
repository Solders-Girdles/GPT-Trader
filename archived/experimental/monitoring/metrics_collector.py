"""In-memory metrics collector with optional demo sampling support."""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque
from collections.abc import Callable


@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    name: str
    points: Deque[MetricPoint]
    max_points: int = 1000

    def add_point(self, value: float, tags: dict[str, str] | None = None) -> None:
        point = MetricPoint(datetime.now(), value, tags or {})
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points.popleft()

    def get_stats(self) -> dict[str, float | int]:
        if not self.points:
            return {}
        values = [p.value for p in self.points]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }


class MetricsCollector:
    def __init__(
        self,
        *,
        system_sampler: Callable[[MetricsCollector], None] | None = None,
    ) -> None:
        self.metrics: dict[str, MetricSeries] = {}
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = {}
        self.histograms: defaultdict[str, list[float]] = defaultdict(list)
        self.timers: dict[str, float] = {}
        self._lock = threading.Lock()
        self.collection_interval = 60
        self._collection_thread: threading.Thread | None = None
        self._running = False
        self._last_collection = datetime.now()
        self._system_sampler = system_sampler

    def start_collection(self) -> None:
        if self._running:
            return
        self._running = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()

    def stop_collection(self) -> None:
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)

    def _collect_loop(self) -> None:
        while self._running:
            try:
                self.collect_system_metrics()
                self._last_collection = datetime.now()
            except Exception:
                pass
            time.sleep(self.collection_interval)

    def collect_system_metrics(self) -> None:
        if self._system_sampler is None:
            return
        self._system_sampler(self)

    def record_counter(self, name: str, increment: int = 1) -> None:
        with self._lock:
            self.counters[name] += increment
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(name, deque(maxlen=1000))
            self.metrics[name].add_point(float(self.counters[name]))

    def record_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(name, deque(maxlen=1000))
            self.metrics[name].add_point(value)

    def record_histogram(self, name: str, value: float) -> None:
        with self._lock:
            self.histograms[name].append(value)
            if len(self.histograms[name]) > 10000:
                self.histograms[name] = self.histograms[name][-10000:]

    def start_timer(self, name: str) -> str:
        timer_id = f"{name}_{int(time.time() * 1_000_000)}"
        with self._lock:
            self.timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str) -> float:
        with self._lock:
            if timer_id in self.timers:
                start_time = self.timers.pop(timer_id)
                duration = time.time() - start_time
                metric_name = timer_id.rsplit("_", 1)[0]
                self.record_histogram(f"{metric_name}.duration_ms", duration * 1000)
                return duration
        return 0.0

    def record_trading_metrics(
        self, trades_executed: int, pnl: float, portfolio_value: float
    ) -> None:
        self.record_counter("trading.trades_executed", trades_executed)
        self.record_gauge("trading.portfolio_value", portfolio_value)
        self.record_gauge("trading.pnl", pnl)
        self.record_histogram("trading.trade_pnl", pnl)

    def record_slice_performance(
        self, slice_name: str, execution_time_ms: float, success: bool
    ) -> None:
        self.record_histogram(f"slices.{slice_name}.execution_time_ms", execution_time_ms)
        self.record_counter(f"slices.{slice_name}.executions")
        if success:
            self.record_counter(f"slices.{slice_name}.successes")
        else:
            self.record_counter(f"slices.{slice_name}.failures")

    def get_metrics_summary(self) -> dict[str, Any]:
        with self._lock:
            summary: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "collection_interval": self.collection_interval,
                "last_collection": self._last_collection.isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "series_stats": {},
            }
            for name, values in self.histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                    }
            for name, series in self.metrics.items():
                summary["series_stats"][name] = series.get_stats()
            return summary

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_metrics(self, window_minutes: int = 60) -> dict[str, list[dict[str, Any]]]:
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        exported: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for name, series in self.metrics.items():
                exported[name] = []
                for point in series.points:
                    if point.timestamp > cutoff:
                        exported[name].append(
                            {
                                "timestamp": point.timestamp.isoformat(),
                                "value": point.value,
                                "tags": point.tags,
                            }
                        )
        return exported

    def get_health_status(self) -> dict[str, Any]:
        with self._lock:
            total_errors = sum(
                count for key, count in self.counters.items() if "error" in key or "failure" in key
            )
            total_successes = sum(count for key, count in self.counters.items() if "success" in key)
            return {
                "status": "healthy" if total_errors < 10 else "degraded",
                "total_errors": total_errors,
                "total_successes": total_successes,
                "error_rate": total_errors / max(total_successes + total_errors, 1),
                "uptime_seconds": time.time(),
                "last_collection": self._last_collection.isoformat(),
                "active_timers": len(self.timers),
            }

    def reset_counters(self) -> None:
        with self._lock:
            self.counters.clear()

    def reset_all(self) -> None:
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.metrics.clear()
            self.timers.clear()


def demo_system_sampler(collector: MetricsCollector) -> None:
    collector.record_gauge("system.health.status", 1.0)
    collector.record_gauge("system.uptime_seconds", time.time())
    collector.record_gauge("slices.available_count", 11)
    for name in (
        "backtest",
        "paper_trade",
        "analyze",
        "optimize",
        "live_trade",
        "monitor",
        "data",
        "ml_strategy",
        "market_regime",
        "position_sizing",
        "adaptive_portfolio",
    ):
        collector.record_gauge(f"slices.{name}.status", 1.0)
    agent_counts = {"available_count": 45, "custom_count": 21, "builtin_count": 24}
    collector.record_gauge("agents.available_count", float(agent_counts["available_count"]))
    collector.record_gauge("agents.custom_count", float(agent_counts["custom_count"]))
    collector.record_gauge("agents.builtin_count", float(agent_counts["builtin_count"]))
    collector.record_histogram("performance.slice_load_time_ms", 50.0)
    collector.record_histogram("performance.api_response_time_ms", 100.0)


_collector: MetricsCollector | None = None
_demo_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
        _collector.start_collection()
    return _collector


def get_demo_metrics_collector() -> MetricsCollector:
    global _demo_collector
    if _demo_collector is None:
        _demo_collector = MetricsCollector(system_sampler=demo_system_sampler)
        _demo_collector.start_collection()
    return _demo_collector


def record_counter(name: str, increment: int = 1) -> None:
    get_metrics_collector().record_counter(name, increment)


def record_gauge(name: str, value: float) -> None:
    get_metrics_collector().record_gauge(name, value)


def record_histogram(name: str, value: float) -> None:
    get_metrics_collector().record_histogram(name, value)


def start_timer(name: str) -> str:
    return get_metrics_collector().start_timer(name)


def stop_timer(timer_id: str) -> float:
    return get_metrics_collector().stop_timer(timer_id)
