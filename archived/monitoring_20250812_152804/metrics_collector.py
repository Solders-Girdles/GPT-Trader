"""
System metrics collection and monitoring.

Provides comprehensive metrics collection for:
- Performance metrics
- Business metrics
- System health metrics
- Custom application metrics
"""

import json
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import psutil


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Statistical summary
    TIMER = "timer"          # Timing measurements


@dataclass
class Metric:
    """Individual metric data."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class MetricSummary:
    """Statistical summary of metric values."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    p50: float  # median
    p95: float
    p99: float
    stddev: float


class MetricsCollector:
    """Collect and manage system metrics."""

    def __init__(self,
                 namespace: str = "gpt_trader",
                 flush_interval: int = 60,
                 max_metrics_age: int = 3600):
        """
        Initialize metrics collector.

        Args:
            namespace: Metric namespace prefix
            flush_interval: Seconds between metric flushes
            max_metrics_age: Maximum age of metrics in seconds
        """
        self.namespace = namespace
        self.flush_interval = flush_interval
        self.max_metrics_age = max_metrics_age

        # Metric storage
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
        self._timers = defaultdict(list)

        # Metric metadata
        self._descriptions = {}
        self._units = {}
        self._labels = defaultdict(dict)

        # Time series data
        self._time_series = defaultdict(lambda: deque(maxlen=1000))

        # Thread safety
        self._lock = threading.RLock()

        # Background flusher
        self._stop_event = threading.Event()
        self._flush_thread = None

        # Metric callbacks
        self._callbacks = []

        # System metrics
        self._process = psutil.Process(os.getpid())

    def start(self):
        """Start background metric collection."""
        if self._flush_thread is None:
            self._flush_thread = threading.Thread(
                target=self._background_flush,
                daemon=True
            )
            self._flush_thread.start()

    def stop(self):
        """Stop background metric collection."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5)

    def _background_flush(self):
        """Background thread to flush metrics."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self.flush()

    # Counter operations
    def increment(self, name: str, value: float = 1.0,
                 labels: dict[str, str] | None = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._record_time_series(key, self._counters[key])

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0.0)

    # Gauge operations
    def gauge(self, name: str, value: float,
              labels: dict[str, str] | None = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._record_time_series(key, value)

    def gauge_increment(self, name: str, value: float = 1.0,
                       labels: dict[str, str] | None = None):
        """Increment a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = self._gauges.get(key, 0.0) + value
            self._record_time_series(key, self._gauges[key])

    # Histogram operations
    def histogram(self, name: str, value: float,
                 labels: dict[str, str] | None = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

            # Limit histogram size
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-5000:]

    # Timer operations
    def timer(self, name: str, labels: dict[str, str] | None = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)

    def record_time(self, name: str, duration: float,
                   labels: dict[str, str] | None = None):
        """Record a timing measurement."""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self._timers:
                self._timers[key] = []
            self._timers[key].append(duration)

            # Limit timer size
            if len(self._timers[key]) > 10000:
                self._timers[key] = self._timers[key][-5000:]

    # Metadata operations
    def describe(self, name: str, description: str, unit: str = ""):
        """Add description and unit to a metric."""
        with self._lock:
            self._descriptions[name] = description
            self._units[name] = unit

    # Time series operations
    def _record_time_series(self, key: str, value: float):
        """Record value in time series."""
        timestamp = datetime.now()
        self._time_series[key].append((timestamp, value))

    def get_time_series(self, name: str,
                        labels: dict[str, str] | None = None,
                        last_n: int | None = None) -> list[tuple[datetime, float]]:
        """Get time series data for a metric."""
        with self._lock:
            key = self._make_key(name, labels)
            series = list(self._time_series.get(key, []))

            if last_n:
                return series[-last_n:]
            return series

    # Summary statistics
    def get_summary(self, name: str,
                   labels: dict[str, str] | None = None) -> MetricSummary | None:
        """Get summary statistics for histogram or timer."""
        with self._lock:
            key = self._make_key(name, labels)

            # Check histograms and timers
            values = self._histograms.get(key) or self._timers.get(key)

            if not values:
                return None

            import numpy as np
            values_array = np.array(values)

            return MetricSummary(
                count=len(values),
                sum=float(np.sum(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                mean=float(np.mean(values_array)),
                p50=float(np.percentile(values_array, 50)),
                p95=float(np.percentile(values_array, 95)),
                p99=float(np.percentile(values_array, 99)),
                stddev=float(np.std(values_array))
            )

    # System metrics
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.gauge("system.cpu.percent", cpu_percent)

        cpu_count = psutil.cpu_count()
        self.gauge("system.cpu.count", cpu_count)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.gauge("system.memory.used", memory.used / (1024 * 1024))  # MB
        self.gauge("system.memory.available", memory.available / (1024 * 1024))
        self.gauge("system.memory.percent", memory.percent)

        # Disk metrics
        disk = psutil.disk_usage('/')
        self.gauge("system.disk.used", disk.used / (1024 * 1024 * 1024))  # GB
        self.gauge("system.disk.free", disk.free / (1024 * 1024 * 1024))
        self.gauge("system.disk.percent", disk.percent)

        # Process metrics
        process_info = self._process.memory_info()
        self.gauge("process.memory.rss", process_info.rss / (1024 * 1024))  # MB
        self.gauge("process.memory.vms", process_info.vms / (1024 * 1024))

        self.gauge("process.cpu.percent", self._process.cpu_percent())
        self.gauge("process.threads", self._process.num_threads())

        # Network metrics (if available)
        try:
            net_io = psutil.net_io_counters()
            self.counter("system.network.bytes_sent", net_io.bytes_sent)
            self.counter("system.network.bytes_recv", net_io.bytes_recv)
            self.counter("system.network.packets_sent", net_io.packets_sent)
            self.counter("system.network.packets_recv", net_io.packets_recv)
        except:
            pass

    # Application metrics
    def collect_application_metrics(self):
        """Collect application-specific metrics."""
        # Override in subclass for specific metrics
        pass

    # Export and reporting
    def get_all_metrics(self) -> list[Metric]:
        """Get all current metrics."""
        metrics = []
        timestamp = datetime.now()

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.COUNTER,
                    value=value,
                    timestamp=timestamp,
                    labels=labels,
                    unit=self._units.get(name, ""),
                    description=self._descriptions.get(name, "")
                ))

            # Gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.GAUGE,
                    value=value,
                    timestamp=timestamp,
                    labels=labels,
                    unit=self._units.get(name, ""),
                    description=self._descriptions.get(name, "")
                ))

            # Histograms (as summaries)
            for key, values in self._histograms.items():
                if values:
                    name, labels = self._parse_key(key)
                    summary = self.get_summary(name, labels)
                    if summary:
                        metrics.append(Metric(
                            name=f"{name}.mean",
                            type=MetricType.SUMMARY,
                            value=summary.mean,
                            timestamp=timestamp,
                            labels=labels,
                            unit=self._units.get(name, ""),
                            description=self._descriptions.get(name, "")
                        ))

            # Timers (as summaries)
            for key, values in self._timers.items():
                if values:
                    name, labels = self._parse_key(key)
                    summary = self.get_summary(name, labels)
                    if summary:
                        metrics.append(Metric(
                            name=f"{name}.mean",
                            type=MetricType.TIMER,
                            value=summary.mean,
                            timestamp=timestamp,
                            labels=labels,
                            unit="seconds",
                            description=self._descriptions.get(name, "")
                        ))

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for metric in self.get_all_metrics():
            # Format labels
            if metric.labels:
                labels_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
                labels_str = f"{{{labels_str}}}"
            else:
                labels_str = ""

            # Format metric line
            metric_name = f"{self.namespace}_{metric.name}".replace(".", "_")

            # Add help and type comments
            if metric.description:
                lines.append(f"# HELP {metric_name} {metric.description}")
            lines.append(f"# TYPE {metric_name} {metric.type.value}")

            # Add metric value
            lines.append(f"{metric_name}{labels_str} {metric.value}")

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics in JSON format."""
        metrics_data = []

        for metric in self.get_all_metrics():
            metrics_data.append({
                "name": metric.name,
                "type": metric.type.value,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "labels": metric.labels,
                "unit": metric.unit,
                "description": metric.description
            })

        return json.dumps(metrics_data, indent=2)

    def flush(self):
        """Flush metrics to registered callbacks."""
        metrics = self.get_all_metrics()

        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                print(f"Error in metrics callback: {e}")

    def register_callback(self, callback: Callable[[list[Metric]], None]):
        """Register a callback for metric updates."""
        self._callbacks.append(callback)

    # Helper methods
    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create unique key for metric with labels."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}:{label_str}"
        return name

    def _parse_key(self, key: str) -> tuple[str, dict[str, str]]:
        """Parse metric key into name and labels."""
        if ":" in key:
            name, label_str = key.split(":", 1)
            labels = {}
            for item in label_str.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    labels[k] = v
            return name, labels
        return key, {}


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str,
                 labels: dict[str, str] | None = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            self.collector.record_time(self.name, duration, self.labels)


class ApplicationMetrics(MetricsCollector):
    """Application-specific metrics for GPT-Trader."""

    def collect_trading_metrics(self, portfolio_value: float = None,
                               positions: int = None,
                               daily_pnl: float = None):
        """Collect trading-specific metrics."""
        if portfolio_value is not None:
            self.gauge("trading.portfolio.value", portfolio_value)

        if positions is not None:
            self.gauge("trading.positions.count", positions)

        if daily_pnl is not None:
            self.gauge("trading.pnl.daily", daily_pnl)

    def collect_backtest_metrics(self, backtest_id: str,
                                sharpe_ratio: float = None,
                                max_drawdown: float = None,
                                total_return: float = None):
        """Collect backtesting metrics."""
        labels = {"backtest_id": backtest_id}

        if sharpe_ratio is not None:
            self.gauge("backtest.sharpe_ratio", sharpe_ratio, labels)

        if max_drawdown is not None:
            self.gauge("backtest.max_drawdown", max_drawdown, labels)

        if total_return is not None:
            self.gauge("backtest.total_return", total_return, labels)

    def collect_strategy_metrics(self, strategy_name: str,
                                signals_generated: int = None,
                                win_rate: float = None):
        """Collect strategy performance metrics."""
        labels = {"strategy": strategy_name}

        if signals_generated is not None:
            self.counter("strategy.signals.generated", signals_generated, labels)

        if win_rate is not None:
            self.gauge("strategy.win_rate", win_rate, labels)


def demo_metrics_collection():
    """Demonstrate metrics collection."""
    print("Metrics Collection Demo")
    print("=" * 50)

    # Create metrics collector
    metrics = ApplicationMetrics(namespace="gpt_trader")
    metrics.start()

    # Describe metrics
    metrics.describe("api.requests", "Number of API requests", "count")
    metrics.describe("api.latency", "API request latency", "seconds")

    # Simulate some operations
    print("\nSimulating application operations...")

    for i in range(10):
        # API request
        metrics.increment("api.requests", labels={"endpoint": "/data"})

        # Timing
        with metrics.timer("api.latency", labels={"endpoint": "/data"}):
            time.sleep(0.01 * (i % 3 + 1))  # Variable latency

        # Business metrics
        metrics.gauge("trading.portfolio.value", 100000 + i * 1000)
        metrics.histogram("order.size", 100 * (i + 1))

    # Collect system metrics
    metrics.collect_system_metrics()

    # Collect trading metrics
    metrics.collect_trading_metrics(
        portfolio_value=105000,
        positions=5,
        daily_pnl=500
    )

    # Get summaries
    print("\nMetric Summaries:")

    latency_summary = metrics.get_summary("api.latency", {"endpoint": "/data"})
    if latency_summary:
        print(f"  API Latency: mean={latency_summary.mean:.3f}s, "
              f"p95={latency_summary.p95:.3f}s")

    order_summary = metrics.get_summary("order.size")
    if order_summary:
        print(f"  Order Size: mean={order_summary.mean:.1f}, "
              f"max={order_summary.max:.1f}")

    # Export metrics
    print("\nPrometheus Format Sample:")
    prometheus_output = metrics.export_prometheus()
    print(prometheus_output[:500] + "...")

    # Get time series
    print("\nTime Series Data:")
    series = metrics.get_time_series("trading.portfolio.value", last_n=5)
    for timestamp, value in series:
        print(f"  {timestamp.strftime('%H:%M:%S')}: ${value:,.0f}")

    # Stop collection
    metrics.stop()

    print("\n✓ Metrics collection demo complete")

    return metrics


if __name__ == "__main__":
    metrics = demo_metrics_collection()
