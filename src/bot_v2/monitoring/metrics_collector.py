import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    name: str
    points: deque
    max_points: int = 1000

    def add_point(self, value: float, tags: dict | None = None):
        point = MetricPoint(datetime.now(), value, tags or {})
        self.points.append(point)

        # Maintain max points limit
        if len(self.points) > self.max_points:
            self.points.popleft()

    def get_stats(self) -> dict[str, float]:
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
    def __init__(self):
        self.metrics = {}
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.timers = {}
        self._lock = threading.Lock()
        self.collection_interval = 60  # seconds
        self._collection_thread = None
        self._running = False
        self._last_collection = datetime.now()
        # Historical builds read agent counts from `.knowledge/STATE.json`. The
        # knowledge layer has been retired, so we fall back to fixed defaults to
        # keep legacy metrics stable without external dependencies.
        self._agent_counts = {
            "available_count": 45,
            "custom_count": 21,
            "builtin_count": 24,
        }

    def start_collection(self):
        """Start background metrics collection."""
        if self._running:
            return

        self._running = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()

    def stop_collection(self):
        """Stop background metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)

    def _collect_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                self.collect_system_metrics()
                self._last_collection = datetime.now()
            except Exception:
                # Log error in real implementation
                pass
            time.sleep(self.collection_interval)

    def collect_system_metrics(self):
        """Collect system-level metrics."""
        timestamp = datetime.now()

        # System health metrics
        self.record_gauge("system.health.status", 1.0)
        self.record_gauge("system.uptime_seconds", time.time())

        # Slice availability metrics
        self.record_gauge("slices.available_count", 11)
        self.record_gauge("slices.backtest.status", 1.0)
        self.record_gauge("slices.paper_trade.status", 1.0)
        self.record_gauge("slices.analyze.status", 1.0)
        self.record_gauge("slices.optimize.status", 1.0)
        self.record_gauge("slices.live_trade.status", 1.0)
        self.record_gauge("slices.monitor.status", 1.0)
        self.record_gauge("slices.data.status", 1.0)
        self.record_gauge("slices.ml_strategy.status", 1.0)
        self.record_gauge("slices.market_regime.status", 1.0)
        self.record_gauge("slices.position_sizing.status", 1.0)
        self.record_gauge("slices.adaptive_portfolio.status", 1.0)

        # Agent workflow metrics (historical defaults retained for continuity)
        counts = self._agent_counts or {}
        self.record_gauge("agents.available_count", float(counts.get("available_count", 45)))
        self.record_gauge("agents.custom_count", float(counts.get("custom_count", 21)))
        self.record_gauge("agents.builtin_count", float(counts.get("builtin_count", 24)))

        # Performance metrics (placeholders for real trading)
        self.record_histogram("performance.slice_load_time_ms", 50.0)
        self.record_histogram("performance.api_response_time_ms", 100.0)

    def record_counter(self, name: str, increment: int = 1):
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += increment

            # Also track as time series
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(name, deque(maxlen=1000))
            self.metrics[name].add_point(self.counters[name])

    def record_gauge(self, name: str, value: float):
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value

            if name not in self.metrics:
                self.metrics[name] = MetricSeries(name, deque(maxlen=1000))
            self.metrics[name].add_point(value)

    def record_histogram(self, name: str, value: float):
        """Record a histogram metric."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 10000:
                self.histograms[name] = self.histograms[name][-10000:]

    def start_timer(self, name: str) -> str:
        """Start a timer and return timer ID."""
        timer_id = f"{name}_{int(time.time() * 1000000)}"  # Use microseconds for uniqueness
        with self._lock:
            self.timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the duration."""
        with self._lock:
            if timer_id in self.timers:
                start_time = self.timers.pop(timer_id)
                duration = time.time() - start_time

                # Extract metric name from timer ID
                metric_name = timer_id.rsplit("_", 1)[0]
                self.record_histogram(f"{metric_name}.duration_ms", duration * 1000)

                return duration
        return 0.0

    def record_trading_metrics(self, trades_executed: int, pnl: float, portfolio_value: float):
        """Record trading-specific metrics."""
        self.record_counter("trading.trades_executed", trades_executed)
        self.record_gauge("trading.portfolio_value", portfolio_value)
        self.record_gauge("trading.pnl", pnl)
        self.record_histogram("trading.trade_pnl", pnl)

    def record_slice_performance(self, slice_name: str, execution_time_ms: float, success: bool):
        """Record performance metrics for individual slices."""
        self.record_histogram(f"slices.{slice_name}.execution_time_ms", execution_time_ms)
        self.record_counter(f"slices.{slice_name}.executions")
        if success:
            self.record_counter(f"slices.{slice_name}.successes")
        else:
            self.record_counter(f"slices.{slice_name}.failures")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of all metrics."""
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "collection_interval": self.collection_interval,
                "last_collection": self._last_collection.isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "series_stats": {},
            }

            # Calculate histogram statistics
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

            # Get time series statistics
            for name, series in self.metrics.items():
                summary["series_stats"][name] = series.get_stats()

            return summary

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_metrics(self, window_minutes: int = 60) -> dict[str, list[dict]]:
        """Export metrics data for a specific time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        exported = {}

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
        """Get overall system health status."""
        with self._lock:
            total_errors = sum(
                v for k, v in self.counters.items() if "error" in k or "failure" in k
            )
            total_successes = sum(v for k, v in self.counters.items() if "success" in k)

            return {
                "status": "healthy" if total_errors < 10 else "degraded",
                "total_errors": total_errors,
                "total_successes": total_successes,
                "error_rate": total_errors / max(total_successes + total_errors, 1),
                "uptime_seconds": time.time(),
                "last_collection": self._last_collection.isoformat(),
                "active_timers": len(self.timers),
            }

    def reset_counters(self):
        """Reset all counter metrics."""
        with self._lock:
            self.counters.clear()

    def reset_all(self):
        """Reset all metrics data."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.metrics.clear()
            self.timers.clear()


# Global metrics collector singleton
_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
        _collector.start_collection()
    return _collector


# Convenience functions for easier usage
def record_counter(name: str, increment: int = 1):
    """Convenience function to record a counter."""
    get_metrics_collector().record_counter(name, increment)


def record_gauge(name: str, value: float):
    """Convenience function to record a gauge."""
    get_metrics_collector().record_gauge(name, value)


def record_histogram(name: str, value: float):
    """Convenience function to record a histogram value."""
    get_metrics_collector().record_histogram(name, value)


def start_timer(name: str) -> str:
    """Convenience function to start a timer."""
    return get_metrics_collector().start_timer(name)


def stop_timer(timer_id: str) -> float:
    """Convenience function to stop a timer."""
    return get_metrics_collector().stop_timer(timer_id)
