"""
GPT-Trader Comprehensive Metrics Collection System

Advanced performance monitoring and analytics providing:
- Real-time metrics collection with minimal performance impact
- Multi-dimensional metrics with tags and labels
- Time-series data aggregation and analysis
- Custom metric types (counters, gauges, histograms, summaries)
- Metric exporters for various monitoring systems (Prometheus, InfluxDB, etc.)
- Performance profiling and bottleneck identification
- Business metrics tracking (trading performance, risk metrics)
- System resource monitoring and alerts

This system provides enterprise-grade observability for all GPT-Trader components
with minimal overhead and maximum insight into system behavior.
"""

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from .base import BaseComponent, ComponentConfig, HealthStatus
from .concurrency import get_concurrency_manager, schedule_recurring_task
from .error_handling import report_error

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Current value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values with buckets
    SUMMARY = "summary"  # Distribution with quantiles
    TIMER = "timer"  # Time duration measurements


class MetricUnit(Enum):
    """Metric units"""

    NONE = ""
    BYTES = "bytes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"
    PERCENT = "percent"
    COUNT = "count"
    RATE_PER_SECOND = "rate_per_second"
    DOLLARS = "dollars"


@dataclass
class MetricLabels:
    """Metric labels for multi-dimensional metrics"""

    labels: dict[str, str] = field(default_factory=dict)

    def add(self, key: str, value: str) -> "MetricLabels":
        """Add a label"""
        self.labels[key] = str(value)
        return self

    def get_key(self) -> str:
        """Get sorted key for labels"""
        if not self.labels:
            return ""

        sorted_items = sorted(self.labels.items())
        return ",".join([f"{k}={v}" for k, v in sorted_items])

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other):
        return isinstance(other, MetricLabels) and self.labels == other.labels


@dataclass
class MetricMetadata:
    """Metadata for metrics"""

    name: str
    metric_type: MetricType
    unit: MetricUnit = MetricUnit.NONE
    description: str = ""
    component_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.metric_type.value} metric: {self.name}"


class MetricValue:
    """Base class for metric values"""

    def __init__(self, metadata: MetricMetadata, labels: MetricLabels | None = None) -> None:
        self.metadata = metadata
        self.labels = labels or MetricLabels()
        self.last_updated = datetime.now()
        self.sample_count = 0

    def get_export_data(self) -> dict[str, Any]:
        """Get data for export to external systems"""
        return {
            "name": self.metadata.name,
            "type": self.metadata.metric_type.value,
            "unit": self.metadata.unit.value,
            "labels": self.labels.labels,
            "last_updated": self.last_updated.isoformat(),
            "sample_count": self.sample_count,
        }


class CounterMetric(MetricValue):
    """Counter metric - monotonically increasing value"""

    def __init__(self, metadata: MetricMetadata, labels: MetricLabels | None = None) -> None:
        super().__init__(metadata, labels)
        self.value = 0.0
        self.lock = threading.RLock()

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter"""
        if amount < 0:
            raise ValueError("Counter increment must be non-negative")

        with self.lock:
            self.value += amount
            self.last_updated = datetime.now()
            self.sample_count += 1

    def get_value(self) -> float:
        """Get current counter value"""
        with self.lock:
            return self.value

    def reset(self) -> None:
        """Reset counter to zero"""
        with self.lock:
            self.value = 0.0
            self.sample_count = 0

    def get_export_data(self) -> dict[str, Any]:
        data = super().get_export_data()
        data["value"] = self.get_value()
        return data


class GaugeMetric(MetricValue):
    """Gauge metric - current value that can increase or decrease"""

    def __init__(self, metadata: MetricMetadata, labels: MetricLabels | None = None) -> None:
        super().__init__(metadata, labels)
        self.value = 0.0
        self.lock = threading.RLock()

    def set(self, value: float) -> None:
        """Set gauge value"""
        with self.lock:
            self.value = value
            self.last_updated = datetime.now()
            self.sample_count += 1

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge value"""
        with self.lock:
            self.value += amount
            self.last_updated = datetime.now()
            self.sample_count += 1

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge value"""
        self.increment(-amount)

    def get_value(self) -> float:
        """Get current gauge value"""
        with self.lock:
            return self.value

    def get_export_data(self) -> dict[str, Any]:
        data = super().get_export_data()
        data["value"] = self.get_value()
        return data


@dataclass
class HistogramBucket:
    """Histogram bucket definition"""

    upper_bound: float
    count: int = 0


class HistogramMetric(MetricValue):
    """Histogram metric - distribution of values with configurable buckets"""

    # Default buckets suitable for most timing measurements (milliseconds)
    DEFAULT_BUCKETS = [
        0.1,
        0.5,
        1,
        2.5,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
        10000,
        float("inf"),
    ]

    def __init__(
        self,
        metadata: MetricMetadata,
        labels: MetricLabels | None = None,
        buckets: list[float] | None = None,
    ) -> None:
        super().__init__(metadata, labels)

        bucket_bounds = buckets or self.DEFAULT_BUCKETS
        self.buckets = [HistogramBucket(bound) for bound in sorted(bucket_bounds)]

        self.total_count = 0
        self.total_sum = 0.0
        self.lock = threading.RLock()

    def observe(self, value: float) -> None:
        """Observe a value"""
        with self.lock:
            self.total_count += 1
            self.total_sum += value
            self.last_updated = datetime.now()
            self.sample_count += 1

            # Update buckets
            for bucket in self.buckets:
                if value <= bucket.upper_bound:
                    bucket.count += 1

    def get_count(self) -> int:
        """Get total observation count"""
        with self.lock:
            return self.total_count

    def get_sum(self) -> float:
        """Get sum of all observed values"""
        with self.lock:
            return self.total_sum

    def get_average(self) -> float:
        """Get average of observed values"""
        with self.lock:
            return self.total_sum / max(1, self.total_count)

    def get_bucket_counts(self) -> list[tuple[float, int]]:
        """Get bucket upper bounds and counts"""
        with self.lock:
            return [(bucket.upper_bound, bucket.count) for bucket in self.buckets]

    def get_percentile(self, percentile: float) -> float:
        """Estimate percentile value"""
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")

        with self.lock:
            if self.total_count == 0:
                return 0.0

            target_count = (percentile / 100) * self.total_count
            cumulative_count = 0

            for i, bucket in enumerate(self.buckets):
                cumulative_count += bucket.count
                if cumulative_count >= target_count:
                    # Linear interpolation within bucket
                    if i == 0:
                        return bucket.upper_bound

                    prev_bound = 0 if i == 0 else self.buckets[i - 1].upper_bound
                    prev_count = cumulative_count - bucket.count

                    if bucket.count == 0:
                        return prev_bound

                    # Linear interpolation
                    ratio = (target_count - prev_count) / bucket.count
                    return prev_bound + ratio * (bucket.upper_bound - prev_bound)

            return self.buckets[-1].upper_bound

    def get_export_data(self) -> dict[str, Any]:
        data = super().get_export_data()
        data.update(
            {
                "count": self.get_count(),
                "sum": self.get_sum(),
                "average": self.get_average(),
                "buckets": self.get_bucket_counts(),
                "p50": self.get_percentile(50),
                "p95": self.get_percentile(95),
                "p99": self.get_percentile(99),
            }
        )
        return data


class SummaryMetric(MetricValue):
    """Summary metric - distribution with sliding window quantiles"""

    def __init__(
        self,
        metadata: MetricMetadata,
        labels: MetricLabels | None = None,
        window_size: int = 1000,
        max_age: timedelta = timedelta(minutes=5),
    ) -> None:
        super().__init__(metadata, labels)

        self.window_size = window_size
        self.max_age = max_age
        self.observations = deque(maxlen=window_size)
        self.total_count = 0
        self.total_sum = 0.0
        self.lock = threading.RLock()

    def observe(self, value: float) -> None:
        """Observe a value"""
        with self.lock:
            timestamp = time.time()
            self.observations.append((timestamp, value))
            self.total_count += 1
            self.total_sum += value
            self.last_updated = datetime.now()
            self.sample_count += 1

            # Remove old observations
            self._clean_old_observations()

    def _clean_old_observations(self) -> None:
        """Remove observations older than max_age"""
        if not self.max_age:
            return

        cutoff_time = time.time() - self.max_age.total_seconds()
        while self.observations and self.observations[0][0] < cutoff_time:
            self.observations.popleft()

    def get_count(self) -> int:
        """Get total observation count"""
        with self.lock:
            return self.total_count

    def get_sum(self) -> float:
        """Get sum of all observed values"""
        with self.lock:
            return self.total_sum

    def get_quantile(self, quantile: float) -> float:
        """Get quantile value from current window"""
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be between 0 and 1")

        with self.lock:
            self._clean_old_observations()

            if not self.observations:
                return 0.0

            values = [obs[1] for obs in self.observations]
            values.sort()

            if quantile == 0:
                return values[0]
            if quantile == 1:
                return values[-1]

            index = quantile * (len(values) - 1)
            lower_index = int(math.floor(index))
            upper_index = int(math.ceil(index))

            if lower_index == upper_index:
                return values[lower_index]

            # Linear interpolation
            weight = index - lower_index
            return values[lower_index] * (1 - weight) + values[upper_index] * weight

    def get_export_data(self) -> dict[str, Any]:
        data = super().get_export_data()
        with self.lock:
            self._clean_old_observations()
            window_count = len(self.observations)
            window_sum = sum(obs[1] for obs in self.observations)

        data.update(
            {
                "count": self.get_count(),
                "sum": self.get_sum(),
                "window_count": window_count,
                "window_sum": window_sum,
                "window_average": window_sum / max(1, window_count),
                "q0.5": self.get_quantile(0.5),
                "q0.95": self.get_quantile(0.95),
                "q0.99": self.get_quantile(0.99),
            }
        )
        return data


class TimerMetric:
    """Timer metric for measuring execution durations"""

    def __init__(self, histogram: HistogramMetric) -> None:
        self.histogram = histogram
        self.start_time: float | None = None

    def start(self):
        """Start timing"""
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        """Stop timing and record duration"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        self.histogram.observe(duration)
        self.start_time = None
        return duration

    @contextmanager
    def time(self):
        """Context manager for timing code blocks"""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def time_callable(self, func: Callable, *args, **kwargs):
        """Time function execution"""
        with self.time():
            return func(*args, **kwargs)


class MetricsRegistry:
    """Registry for managing metrics"""

    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, MetricValue]] = defaultdict(
            dict
        )  # {metric_name: {label_key: metric}}
        self.metadata: dict[str, MetricMetadata] = {}
        self.lock = threading.RLock()

        logger.debug("Metrics registry initialized")

    def register_counter(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
        component_id: str = "",
        labels: MetricLabels | None = None,
    ) -> CounterMetric:
        """Register a counter metric"""
        metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.COUNTER,
            unit=unit,
            description=description,
            component_id=component_id,
        )

        return self._register_metric(metadata, CounterMetric, labels)

    def register_gauge(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.NONE,
        component_id: str = "",
        labels: MetricLabels | None = None,
    ) -> GaugeMetric:
        """Register a gauge metric"""
        metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.GAUGE,
            unit=unit,
            description=description,
            component_id=component_id,
        )

        return self._register_metric(metadata, GaugeMetric, labels)

    def register_histogram(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        component_id: str = "",
        labels: MetricLabels | None = None,
        buckets: list[float] | None = None,
    ) -> HistogramMetric:
        """Register a histogram metric"""
        metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            unit=unit,
            description=description,
            component_id=component_id,
        )

        return self._register_metric(metadata, HistogramMetric, labels, buckets=buckets)

    def register_summary(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        component_id: str = "",
        labels: MetricLabels | None = None,
    ) -> SummaryMetric:
        """Register a summary metric"""
        metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.SUMMARY,
            unit=unit,
            description=description,
            component_id=component_id,
        )

        return self._register_metric(metadata, SummaryMetric, labels)

    def register_timer(
        self,
        name: str,
        description: str = "",
        component_id: str = "",
        labels: MetricLabels | None = None,
    ) -> TimerMetric:
        """Register a timer metric (histogram-based)"""
        histogram = self.register_histogram(
            name=name,
            description=description or f"Timer: {name}",
            unit=MetricUnit.MILLISECONDS,
            component_id=component_id,
            labels=labels,
        )

        return TimerMetric(histogram)

    def _register_metric(
        self,
        metadata: MetricMetadata,
        metric_class: type,
        labels: MetricLabels | None = None,
        **kwargs,
    ) -> MetricValue:
        """Register a metric with the registry"""
        with self.lock:
            labels = labels or MetricLabels()
            label_key = labels.get_key()

            # Store metadata (only once per metric name)
            if metadata.name not in self.metadata:
                self.metadata[metadata.name] = metadata

            # Check if metric already exists with these labels
            if label_key in self.metrics[metadata.name]:
                return self.metrics[metadata.name][label_key]

            # Create new metric instance
            metric = metric_class(metadata, labels, **kwargs)
            self.metrics[metadata.name][label_key] = metric

            logger.debug(f"Registered {metadata.metric_type.value} metric: {metadata.name}")
            return metric

    def get_metric(self, name: str, labels: MetricLabels | None = None) -> MetricValue | None:
        """Get metric by name and labels"""
        with self.lock:
            if name not in self.metrics:
                return None

            labels = labels or MetricLabels()
            label_key = labels.get_key()

            return self.metrics[name].get(label_key)

    def get_all_metrics(self) -> dict[str, list[MetricValue]]:
        """Get all registered metrics"""
        with self.lock:
            result = {}
            for metric_name, label_dict in self.metrics.items():
                result[metric_name] = list(label_dict.values())
            return result

    def get_metrics_by_component(self, component_id: str) -> dict[str, list[MetricValue]]:
        """Get all metrics for a specific component"""
        with self.lock:
            result = {}
            for metric_name, label_dict in self.metrics.items():
                metadata = self.metadata.get(metric_name)
                if metadata and metadata.component_id == component_id:
                    result[metric_name] = list(label_dict.values())
            return result

    def export_all_metrics(self) -> dict[str, Any]:
        """Export all metrics for external systems"""
        with self.lock:
            exported = {}

            for metric_name, label_dict in self.metrics.items():
                metric_data = []
                for metric in label_dict.values():
                    metric_data.append(metric.get_export_data())

                exported[metric_name] = {
                    "metadata": self.metadata[metric_name].__dict__,
                    "metrics": metric_data,
                }

            return exported

    def clear_metrics(self, pattern: str = "*") -> None:
        """Clear metrics matching pattern"""
        import fnmatch

        with self.lock:
            metrics_to_remove = []

            for metric_name in self.metrics.keys():
                if fnmatch.fnmatch(metric_name, pattern):
                    metrics_to_remove.append(metric_name)

            for metric_name in metrics_to_remove:
                del self.metrics[metric_name]
                if metric_name in self.metadata:
                    del self.metadata[metric_name]

            logger.info(f"Cleared {len(metrics_to_remove)} metrics matching pattern: {pattern}")


class MetricsCollector(BaseComponent):
    """
    Comprehensive metrics collection and management system
    """

    def __init__(self, config: ComponentConfig | None = None) -> None:
        if not config:
            config = ComponentConfig(
                component_id="metrics_collector", component_type="metrics_collector"
            )

        super().__init__(config)

        # Core components
        self.registry = MetricsRegistry()
        self.exporters: list[IMetricExporter] = []

        # System metrics
        self.system_metrics = {}
        self.business_metrics = {}

        # Collection settings
        self.collection_interval = timedelta(seconds=30)
        self.export_interval = timedelta(minutes=1)

        # Initialize built-in metrics
        self._initialize_system_metrics()

        logger.info("Metrics collector initialized")

    def _initialize_component(self) -> None:
        """Initialize metrics collector"""
        # Schedule metrics collection
        schedule_recurring_task(
            task_id="collect_system_metrics",
            function=self._collect_system_metrics,
            interval=self.collection_interval,
            component_id=self.component_id,
        )

        # Schedule metrics export
        schedule_recurring_task(
            task_id="export_metrics",
            function=self._export_metrics,
            interval=self.export_interval,
            component_id=self.component_id,
        )

    def _start_component(self) -> None:
        """Start metrics collection"""
        logger.info("Metrics collection started")

    def _stop_component(self) -> None:
        """Stop metrics collection"""
        # Export final metrics
        self._export_metrics()
        logger.info("Metrics collection stopped")

    def _health_check(self) -> HealthStatus:
        """Check metrics collector health"""
        try:
            # Check if we have recent metrics
            all_metrics = self.registry.get_all_metrics()
            if not all_metrics:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNHEALTHY

    def _initialize_system_metrics(self) -> None:
        """Initialize built-in system metrics"""
        # Performance metrics
        self.system_metrics.update(
            {
                "requests_total": self.registry.register_counter(
                    "requests_total",
                    "Total number of requests processed",
                    unit=MetricUnit.COUNT,
                    component_id=self.component_id,
                ),
                "request_duration": self.registry.register_histogram(
                    "request_duration_milliseconds",
                    "Request processing duration",
                    unit=MetricUnit.MILLISECONDS,
                    component_id=self.component_id,
                ),
                "active_connections": self.registry.register_gauge(
                    "active_connections",
                    "Number of active connections",
                    unit=MetricUnit.COUNT,
                    component_id=self.component_id,
                ),
                "error_rate": self.registry.register_counter(
                    "errors_total",
                    "Total number of errors",
                    unit=MetricUnit.COUNT,
                    component_id=self.component_id,
                ),
                "cache_hit_rate": self.registry.register_gauge(
                    "cache_hit_rate_percent",
                    "Cache hit rate percentage",
                    unit=MetricUnit.PERCENT,
                    component_id=self.component_id,
                ),
                "thread_pool_utilization": self.registry.register_gauge(
                    "thread_pool_utilization_percent",
                    "Thread pool utilization percentage",
                    unit=MetricUnit.PERCENT,
                    component_id=self.component_id,
                ),
                "memory_usage": self.registry.register_gauge(
                    "memory_usage_bytes",
                    "Memory usage in bytes",
                    unit=MetricUnit.BYTES,
                    component_id=self.component_id,
                ),
            }
        )

        # Business metrics
        self.business_metrics.update(
            {
                "trades_executed": self.registry.register_counter(
                    "trades_executed_total",
                    "Total number of trades executed",
                    unit=MetricUnit.COUNT,
                    component_id=self.component_id,
                ),
                "pnl_realized": self.registry.register_gauge(
                    "pnl_realized_dollars",
                    "Realized profit and loss",
                    unit=MetricUnit.DOLLARS,
                    component_id=self.component_id,
                ),
                "pnl_unrealized": self.registry.register_gauge(
                    "pnl_unrealized_dollars",
                    "Unrealized profit and loss",
                    unit=MetricUnit.DOLLARS,
                    component_id=self.component_id,
                ),
                "portfolio_value": self.registry.register_gauge(
                    "portfolio_value_dollars",
                    "Current portfolio value",
                    unit=MetricUnit.DOLLARS,
                    component_id=self.component_id,
                ),
                "risk_var_95": self.registry.register_gauge(
                    "risk_var_95_dollars",
                    "95% Value at Risk",
                    unit=MetricUnit.DOLLARS,
                    component_id=self.component_id,
                ),
                "order_latency": self.registry.register_histogram(
                    "order_latency_milliseconds",
                    "Order execution latency",
                    unit=MetricUnit.MILLISECONDS,
                    component_id=self.component_id,
                ),
            }
        )

    def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics"""
        try:
            # Get system statistics from various components
            concurrency_manager = get_concurrency_manager()
            concurrency_stats = concurrency_manager.get_system_stats()

            # Update system metrics
            self._update_thread_pool_metrics(concurrency_stats.get("thread_pools", {}))
            self._update_memory_metrics()

            # Get cache statistics if available
            try:
                from .caching import get_cache_manager

                cache_manager = get_cache_manager()
                cache_stats = cache_manager.get_system_statistics()
                self._update_cache_metrics(cache_stats.get("global_statistics", {}))
            except ImportError:
                pass  # Cache manager not available

            self.record_operation(success=True)

        except Exception as e:
            logger.error(f"System metrics collection error: {str(e)}")
            self.record_operation(success=False, error_message=str(e))
            report_error(e, component=self.component_id)

    def _update_thread_pool_metrics(self, thread_pool_stats: dict[str, Any]) -> None:
        """Update thread pool utilization metrics"""
        for pool_name, pool_stats in thread_pool_stats.items():
            labels = MetricLabels().add("pool_type", pool_name)

            utilization_metric = self.registry.register_gauge(
                "thread_pool_utilization_percent",
                "Thread pool utilization percentage",
                unit=MetricUnit.PERCENT,
                component_id=self.component_id,
                labels=labels,
            )

            # Calculate utilization
            active_tasks = pool_stats.get("active_tasks", 0)
            max_workers = pool_stats.get("max_workers", 1)
            utilization = (active_tasks / max_workers) * 100

            utilization_metric.set(utilization)

    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            self.system_metrics["memory_usage"].set(memory_info.rss)

        except ImportError:
            # psutil not available, skip memory metrics
            pass

    def _update_cache_metrics(self, cache_stats: dict[str, Any]) -> None:
        """Update cache performance metrics"""
        hit_rate = cache_stats.get("global_hit_rate", 0.0)
        self.system_metrics["cache_hit_rate"].set(hit_rate)

    def _export_metrics(self) -> None:
        """Export metrics to configured exporters"""
        try:
            if not self.exporters:
                return

            exported_data = self.registry.export_all_metrics()

            for exporter in self.exporters:
                try:
                    exporter.export(exported_data)
                except Exception as e:
                    logger.error(f"Metric export error for {exporter.__class__.__name__}: {str(e)}")

        except Exception as e:
            logger.error(f"Metrics export error: {str(e)}")

    def get_registry(self) -> MetricsRegistry:
        """Get metrics registry"""
        return self.registry

    def add_exporter(self, exporter: "IMetricExporter") -> None:
        """Add metric exporter"""
        self.exporters.append(exporter)
        logger.info(f"Added metric exporter: {exporter.__class__.__name__}")

    def record_request(
        self, duration_ms: float, success: bool = True, labels: MetricLabels | None = None
    ) -> None:
        """Record request metrics"""
        # Increment request counter
        request_counter = self.registry.get_metric("requests_total", labels)
        if request_counter:
            request_counter.increment()

        # Record duration
        duration_histogram = self.registry.get_metric("request_duration_milliseconds", labels)
        if duration_histogram:
            duration_histogram.observe(duration_ms)

        # Record errors
        if not success:
            error_counter = self.registry.get_metric("errors_total", labels)
            if error_counter:
                error_counter.increment()

    def record_trade(
        self, symbol: str, quantity: float, price: float, side: str, latency_ms: float
    ) -> None:
        """Record trading metrics"""
        labels = MetricLabels().add("symbol", symbol).add("side", side)

        # Record trade execution
        trade_counter = self.registry.get_metric("trades_executed_total", labels)
        if trade_counter:
            trade_counter.increment()

        # Record order latency
        latency_histogram = self.registry.get_metric("order_latency_milliseconds", labels)
        if latency_histogram:
            latency_histogram.observe(latency_ms)

    def update_portfolio_metrics(
        self, portfolio_value: float, pnl_realized: float, pnl_unrealized: float, var_95: float
    ) -> None:
        """Update portfolio and risk metrics"""
        self.business_metrics["portfolio_value"].set(portfolio_value)
        self.business_metrics["pnl_realized"].set(pnl_realized)
        self.business_metrics["pnl_unrealized"].set(pnl_unrealized)
        self.business_metrics["risk_var_95"].set(var_95)

    def get_metric_summary(self) -> dict[str, Any]:
        """Get summary of all metrics"""
        all_metrics = self.registry.get_all_metrics()

        summary = {
            "total_metrics": sum(len(metrics) for metrics in all_metrics.values()),
            "metric_types": {},
            "components": set(),
        }

        for _metric_name, metrics in all_metrics.items():
            if metrics:
                metric_type = metrics[0].metadata.metric_type.value
                summary["metric_types"][metric_type] = summary["metric_types"].get(
                    metric_type, 0
                ) + len(metrics)

                component_id = metrics[0].metadata.component_id
                if component_id:
                    summary["components"].add(component_id)

        summary["components"] = list(summary["components"])

        return summary


# Metric exporters for external systems


class IMetricExporter(ABC):
    """Interface for metric exporters"""

    @abstractmethod
    def export(self, metrics_data: dict[str, Any]) -> bool:
        """Export metrics data"""
        pass


class LogMetricExporter(IMetricExporter):
    """Export metrics to log files"""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(f"{__name__}.metrics_export")
        self.log_level = log_level

    def export(self, metrics_data: dict[str, Any]) -> bool:
        """Export metrics to logs"""
        try:
            for metric_name, metric_info in metrics_data.items():
                for metric in metric_info["metrics"]:
                    labels_str = ""
                    if metric["labels"]:
                        labels_str = (
                            "{" + ",".join([f"{k}={v}" for k, v in metric["labels"].items()]) + "}"
                        )

                    log_line = f"METRIC {metric_name}{labels_str} = {metric.get('value', 'N/A')} [{metric['type']}]"
                    self.logger.log(self.log_level, log_line)

            return True

        except Exception as e:
            logger.error(f"Log metric export error: {str(e)}")
            return False


class PrometheusMetricExporter(IMetricExporter):
    """Export metrics in Prometheus format"""

    def __init__(self, output_file: str = "metrics.prom") -> None:
        self.output_file = output_file

    def export(self, metrics_data: dict[str, Any]) -> bool:
        """Export metrics in Prometheus format"""
        try:
            with open(self.output_file, "w") as f:
                for metric_name, metric_info in metrics_data.items():
                    metadata = metric_info["metadata"]

                    # Write help and type comments
                    f.write(f"# HELP {metric_name} {metadata['description']}\n")
                    f.write(f"# TYPE {metric_name} {metadata['type']}\n")

                    # Write metric values
                    for metric in metric_info["metrics"]:
                        labels_str = ""
                        if metric["labels"]:
                            label_pairs = [f'{k}="{v}"' for k, v in metric["labels"].items()]
                            labels_str = "{" + ",".join(label_pairs) + "}"

                        if metric["type"] == "histogram":
                            # Export histogram buckets
                            for bound, count in metric.get("buckets", []):
                                bucket_labels = (
                                    labels_str.rstrip("}") + f',le="{bound}"}}'
                                    if labels_str
                                    else f'{{le="{bound}"}}'
                                )
                                f.write(f"{metric_name}_bucket{bucket_labels} {count}\n")

                            f.write(f"{metric_name}_count{labels_str} {metric.get('count', 0)}\n")
                            f.write(f"{metric_name}_sum{labels_str} {metric.get('sum', 0)}\n")
                        else:
                            value = metric.get("value", 0)
                            f.write(f"{metric_name}{labels_str} {value}\n")

                    f.write("\n")

            return True

        except Exception as e:
            logger.error(f"Prometheus export error: {str(e)}")
            return False


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector

    with _metrics_lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector()
            logger.info("Global metrics collector created")

        return _metrics_collector


def get_metrics_registry() -> MetricsRegistry:
    """Get global metrics registry"""
    return get_metrics_collector().get_registry()


# Decorators for automatic metrics collection


def track_execution_time(metric_name: str = None, labels: MetricLabels | None = None):
    """Decorator to track function execution time"""

    def decorator(func: F) -> F:
        nonlocal metric_name
        if not metric_name:
            metric_name = f"{func.__module__}.{func.__name__}_duration"

        registry = get_metrics_registry()
        timer = registry.register_timer(
            metric_name, f"Execution time for {func.__name__}", component_id=func.__module__
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer.time():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def count_calls(metric_name: str = None, labels: MetricLabels | None = None):
    """Decorator to count function calls"""

    def decorator(func: F) -> F:
        nonlocal metric_name
        if not metric_name:
            metric_name = f"{func.__module__}.{func.__name__}_calls"

        registry = get_metrics_registry()
        counter = registry.register_counter(
            metric_name,
            f"Number of calls to {func.__name__}",
            component_id=func.__module__,
            labels=labels,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            counter.increment()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def track_errors(metric_name: str = None, labels: MetricLabels | None = None):
    """Decorator to track function errors"""

    def decorator(func: F) -> F:
        nonlocal metric_name
        if not metric_name:
            metric_name = f"{func.__module__}.{func.__name__}_errors"

        registry = get_metrics_registry()
        error_counter = registry.register_counter(
            metric_name,
            f"Number of errors in {func.__name__}",
            component_id=func.__module__,
            labels=labels,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                error_counter.increment()
                raise

        return wrapper

    return decorator


@contextmanager
def metrics_context(operation: str, component_id: str = ""):
    """Context manager for tracking operation metrics"""
    registry = get_metrics_registry()

    # Create metrics
    timer = registry.register_timer(f"{operation}_duration", component_id=component_id)
    success_counter = registry.register_counter(f"{operation}_success", component_id=component_id)
    error_counter = registry.register_counter(f"{operation}_errors", component_id=component_id)

    try:
        with timer.time():
            yield
        success_counter.increment()
    except Exception:
        error_counter.increment()
        raise
