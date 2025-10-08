"""Performance monitoring and metrics collection utilities."""

from __future__ import annotations

import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TypeVar

from bot_v2.utilities.logging_patterns import get_logger

T = TypeVar("T")

logger = get_logger("performance", component="monitoring")


@dataclass
class PerformanceMetric:
    """Single performance metric data point."""
    
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of metric."""
        tags_str = ",".join(f"{k}={v}" for k, v in self.tags.items())
        tags_part = f"[{tags_str}]" if tags_str else ""
        return f"{self.name}{tags_part}: {self.value:.3f}{self.unit}"


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0.0
    recent_avg: float = 0.0
    
    def update(self, value: float) -> None:
        """Update statistics with new value.
        
        Args:
            value: New metric value
        """
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = self.total / self.count
        
    def __str__(self) -> str:
        """String representation of statistics."""
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
        """Initialize collector.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._lock = threading.RLock()
        
    def record(self, metric: PerformanceMetric) -> None:
        """Record a performance metric.
        
        Args:
            metric: Metric to record
        """
        with self._lock:
            # Add to history
            self._metrics[metric.name].append(metric)
            
            # Update statistics
            self._stats[metric.name].update(metric.value)
            
            # Log significant metrics
            if metric.value > 1.0:  # Log slow operations
                logger.warning(
                    f"Slow operation detected: {metric}",
                    operation=metric.name,
                    duration_ms=metric.value * 1000,
                    **metric.tags
                )
                
    def get_stats(self, metric_name: str) -> PerformanceStats:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Performance statistics
        """
        with self._lock:
            return self._stats[metric_name]
            
    def get_recent_metrics(self, metric_name: str, count: int = 10) -> List[PerformanceMetric]:
        """Get recent metrics for a name.
        
        Args:
            metric_name: Name of metric
            count: Number of recent metrics to return
            
        Returns:
            List of recent metrics
        """
        with self._lock:
            metrics = list(self._metrics[metric_name])
            return metrics[-count:] if metrics else []
            
    def get_all_metric_names(self) -> List[str]:
        """Get all recorded metric names.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._metrics.keys())
            
    def clear(self, metric_name: str | None = None) -> None:
        """Clear metrics.
        
        Args:
            metric_name: Specific metric to clear (None for all)
        """
        with self._lock:
            if metric_name:
                self._metrics[metric_name].clear()
                self._stats[metric_name] = PerformanceStats()
            else:
                self._metrics.clear()
                self._stats.clear()
                
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        with self._lock:
            summary = {}
            for name, stats in self._stats.items():
                if stats.count > 0:
                    summary[name] = {
                        "count": stats.count,
                        "avg": stats.avg,
                        "min": stats.min,
                        "max": stats.max,
                        "total": stats.total,
                        "recent_count": len(self._metrics[name]),
                    }
            return summary


# Global performance collector instance
_global_collector = PerformanceCollector()


def get_collector() -> PerformanceCollector:
    """Get the global performance collector.
    
    Returns:
        Global performance collector instance
    """
    return _global_collector


@contextmanager
def measure_performance(
    operation_name: str,
    tags: Dict[str, str] | None = None,
    collector: PerformanceCollector | None = None,
) -> Any:
    """Context manager for measuring operation performance.
    
    Args:
        operation_name: Name of operation being measured
        tags: Additional tags for the metric
        collector: Collector to use (uses global if None)
        
    Yields:
        None
    """
    start_time = time.time()
    tags = tags or {}
    collector = collector or _global_collector
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        metric = PerformanceMetric(
            name=operation_name,
            value=duration,
            unit="s",
            tags=tags
        )
        collector.record(metric)


def measure_performance_decorator(
    operation_name: str | None = None,
    tags: Dict[str, str] | None = None,
    collector: PerformanceCollector | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for measuring function performance.
    
    Args:
        operation_name: Name for operation (uses function name if None)
        tags: Additional tags for metrics
        collector: Collector to use (uses global if None)
        
    Returns:
        Decorated function
    """
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
        tags: Dict[str, str] | None = None,
        collector: PerformanceCollector | None = None,
    ) -> None:
        """Initialize timer.
        
        Args:
            operation_name: Name of operation
            tags: Additional tags
            collector: Collector to use
        """
        self.operation_name = operation_name
        self.tags = tags or {}
        self.collector = collector or _global_collector
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop the timer and record the metric.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
            
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        metric = PerformanceMetric(
            name=self.operation_name,
            value=duration,
            unit="s",
            tags=self.tags
        )
        self.collector.record(metric)
        
        return duration
        
    def __enter__(self) -> PerformanceTimer:
        """Enter context manager."""
        self.start()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.stop()


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self) -> None:
        """Initialize resource monitor."""
        self._psutil = None
        self._try_import_psutil()
        
    def _try_import_psutil(self) -> None:
        """Try to import psutil for system monitoring."""
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.debug("psutil not available, resource monitoring disabled")
            
    def is_available(self) -> bool:
        """Check if resource monitoring is available.
        
        Returns:
            True if psutil is available
        """
        return self._psutil is not None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if not self.is_available():
            return {}
            
        process = self._psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
        
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage.
        
        Returns:
            Dictionary with CPU statistics
        """
        if not self.is_available():
            return {}
            
        process = self._psutil.Process()
        
        return {
            "cpu_percent": process.cpu_percent(),
            "cpu_count": self._psutil.cpu_count(),
        }
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.
        
        Returns:
            Dictionary with system information
        """
        if not self.is_available():
            return {}
            
        return {
            "cpu_count": self._psutil.cpu_count(),
            "memory_total_gb": self._psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": self._psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "memory_percent": self._psutil.virtual_memory().percent,
        }


# Global resource monitor
_resource_monitor = ResourceMonitor()


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor.
    
    Returns:
        Global resource monitor instance
    """
    return _resource_monitor


class PerformanceProfiler:
    """Profile function performance over time."""
    
    def __init__(self, sample_rate: float = 0.1) -> None:
        """Initialize profiler.
        
        Args:
            sample_rate: Fraction of calls to sample (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._total_times: Dict[str, float] = defaultdict(float)
        
    def should_sample(self) -> bool:
        """Check if current call should be sampled.
        
        Returns:
            True if should sample
        """
        import random
        return random.random() < self.sample_rate
        
    def record_call(self, func_name: str, duration: float) -> None:
        """Record a function call.
        
        Args:
            func_name: Name of function
            duration: Call duration in seconds
        """
        self._call_counts[func_name] += 1
        self._total_times[func_name] += duration
        
    def get_profile_data(self) -> Dict[str, Dict[str, float]]:
        """Get profiling data.
        
        Returns:
            Dictionary with profiling statistics
        """
        profile_data = {}
        for func_name in self._call_counts:
            count = self._call_counts[func_name]
            total_time = self._total_times[func_name]
            avg_time = total_time / count if count > 0 else 0
            
            profile_data[func_name] = {
                "call_count": count,
                "total_time": total_time,
                "avg_time": avg_time,
                "sample_rate": self.sample_rate,
            }
            
        return profile_data


# Global profiler
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler.
    
    Returns:
        Global profiler instance
    """
    return _global_profiler


def profile_performance(
    sample_rate: float = 0.1,
    profiler: PerformanceProfiler | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for profiling function performance.
    
    Args:
        sample_rate: Fraction of calls to sample
        profiler: Profiler to use (uses global if None)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        prof = profiler or _global_profiler
        
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not prof.should_sample():
                return func(*args, **kwargs)
                
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                func_name = f"{func.__module__}.{func.__name__}"
                prof.record_call(func_name, duration)
                
        return wrapper
    return decorator


class PerformanceReporter:
    """Generate performance reports."""
    
    def __init__(
        self,
        collector: PerformanceCollector | None = None,
        resource_monitor: ResourceMonitor | None = None,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        """Initialize reporter.
        
        Args:
            collector: Performance collector to use
            resource_monitor: Resource monitor to use
            profiler: Profiler to use
        """
        self.collector = collector or _global_collector
        self.resource_monitor = resource_monitor or _resource_monitor
        self.profiler = profiler or _global_profiler
        
    def generate_report(self) -> str:
        """Generate comprehensive performance report.
        
        Returns:
            Performance report as string
        """
        lines = []
        lines.append("Performance Report")
        lines.append("=" * 50)
        
        # Performance metrics summary
        lines.append("\nPerformance Metrics:")
        lines.append("-" * 20)
        
        summary = self.collector.get_summary()
        if summary:
            for name, stats in sorted(summary.items()):
                lines.append(f"{name}: {stats}")
        else:
            lines.append("No metrics recorded")
            
        # Resource usage
        lines.append("\nResource Usage:")
        lines.append("-" * 15)
        
        if self.resource_monitor.is_available():
            memory = self.resource_monitor.get_memory_usage()
            cpu = self.resource_monitor.get_cpu_usage()
            
            if memory:
                lines.append(f"Memory: {memory['rss_mb']:.1f}MB RSS, {memory['percent']:.1f}%")
            if cpu:
                lines.append(f"CPU: {cpu['cpu_percent']:.1f}%")
        else:
            lines.append("Resource monitoring not available")
            
        # Profiling data
        lines.append("\nProfiling Data:")
        lines.append("-" * 15)
        
        profile_data = self.profiler.get_profile_data()
        if profile_data:
            for func_name, data in sorted(profile_data.items(), key=lambda x: x[1]['total_time'], reverse=True):
                lines.append(
                    f"{func_name}: {data['call_count']} calls, "
                    f"{data['avg_time']:.3f}s avg, {data['total_time']:.3f}s total"
                )
        else:
            lines.append("No profiling data available")
            
        return "\n".join(lines)
        
    def log_report(self, level: int = 20) -> None:
        """Log the performance report.
        
        Args:
            level: Logging level
        """
        report = self.generate_report()
        logger.log(level, f"\n{report}")
        
    def save_report(self, filepath: str) -> None:
        """Save performance report to file.
        
        Args:
            filepath: Path to save report
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Performance report saved to {filepath}")


# Utility functions for common performance monitoring scenarios

def monitor_trading_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator specifically for trading operations.
    
    Args:
        operation_name: Name of trading operation
        
    Returns:
        Decorated function
    """
    return measure_performance_decorator(
        operation_name=f"trading.{operation_name}",
        tags={"component": "trading"},
    )


def monitor_database_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator specifically for database operations.
    
    Args:
        operation_name: Name of database operation
        
    Returns:
        Decorated function
    """
    return measure_performance_decorator(
        operation_name=f"database.{operation_name}",
        tags={"component": "database"},
    )


def monitor_api_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator specifically for API operations.
    
    Args:
        operation_name: Name of API operation
        
    Returns:
        Decorated function
    """
    return measure_performance_decorator(
        operation_name=f"api.{operation_name}",
        tags={"component": "api"},
    )


def get_performance_health_check() -> Dict[str, Any]:
    """Get performance health check status.
    
    Returns:
        Health check results
    """
    health = {
        "status": "healthy",
        "issues": [],
        "metrics": {},
    }
    
    # Check for slow operations
    summary = _global_collector.get_summary()
    for name, stats in summary.items():
        if stats.avg > 1.0:  # Operations averaging > 1s
            health["issues"].append(f"Slow operation: {name} averaging {stats.avg:.3f}s")
            health["status"] = "degraded"
            
        if stats.max > 5.0:  # Any operation > 5s
            health["issues"].append(f"Very slow operation: {name} peaked at {stats.max:.3f}s")
            health["status"] = "unhealthy"
            
    # Check memory usage
    if _resource_monitor.is_available():
        memory = _resource_monitor.get_memory_usage()
        if memory.get("percent", 0) > 80:
            health["issues"].append(f"High memory usage: {memory['percent']:.1f}%")
            health["status"] = "degraded"
            
        cpu = _resource_monitor.get_cpu_usage()
        if cpu.get("cpu_percent", 0) > 80:
            health["issues"].append(f"High CPU usage: {cpu['cpu_percent']:.1f}%")
            health["status"] = "degraded"
            
    health["metrics"] = {
        "total_metrics": len(summary),
        "memory_usage_mb": memory.get("rss_mb", 0),
        "cpu_usage_percent": cpu.get("cpu_percent", 0),
    }
    
    return health
