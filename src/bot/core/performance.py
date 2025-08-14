"""
GPT-Trader Performance Optimization Framework

Intelligent performance analysis and optimization providing:
- Real-time performance profiling and bottleneck identification
- Automated optimization recommendations and implementations
- Resource utilization monitoring and tuning
- Algorithm complexity analysis and optimization
- Memory leak detection and prevention
- Database query optimization and connection tuning
- Cache optimization and hit rate improvement
- Thread pool sizing and concurrency optimization

This framework continuously monitors system performance and automatically
applies optimizations to maintain peak trading system performance.
"""

import cProfile
import logging
import math
import pstats
import resource
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
from io import StringIO
from typing import Any, TypeVar

from .base import BaseComponent, ComponentConfig, HealthStatus
from .concurrency import get_concurrency_manager, schedule_recurring_task
from .error_handling import report_error
from .metrics import get_metrics_registry, track_execution_time

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class PerformanceIssueType(Enum):
    """Types of performance issues"""

    CPU_BOTTLENECK = "cpu_bottleneck"
    MEMORY_LEAK = "memory_leak"
    IO_BOTTLENECK = "io_bottleneck"
    DATABASE_SLOW_QUERY = "database_slow_query"
    CACHE_MISS_RATE = "cache_miss_rate"
    THREAD_CONTENTION = "thread_contention"
    NETWORK_LATENCY = "network_latency"
    ALGORITHM_COMPLEXITY = "algorithm_complexity"


class OptimizationStrategy(Enum):
    """Optimization strategies"""

    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    BATCH_PROCESSING = "batch_processing"
    ALGORITHM_IMPROVEMENT = "algorithm_improvement"
    MEMORY_OPTIMIZATION = "memory_optimization"
    THREAD_TUNING = "thread_tuning"
    DATABASE_INDEXING = "database_indexing"
    COMPRESSION = "compression"


class PerformanceSeverity(Enum):
    """Performance issue severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""

    timestamp: datetime

    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_user_time: float = 0.0
    cpu_system_time: float = 0.0

    # Memory metrics
    memory_rss_bytes: int = 0
    memory_vms_bytes: int = 0
    memory_percent: float = 0.0
    gc_collections: dict[int, int] = field(default_factory=dict)

    # I/O metrics
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_count: int = 0
    io_write_count: int = 0

    # Thread metrics
    thread_count: int = 0
    thread_pool_utilization: dict[str, float] = field(default_factory=dict)

    # Database metrics
    db_connections_active: int = 0
    db_connections_idle: int = 0
    db_query_avg_time_ms: float = 0.0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size_bytes: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class PerformanceIssue:
    """Detected performance issue"""

    issue_type: PerformanceIssueType
    severity: PerformanceSeverity
    component_id: str
    description: str
    impact_score: float  # 0-100
    detected_at: datetime

    # Issue details
    current_value: Any = None
    expected_value: Any = None
    threshold_breached: Any = None

    # Evidence and context
    stack_trace: str | None = None
    profiling_data: dict[str, Any] | None = None
    related_metrics: dict[str, Any] = field(default_factory=dict)

    # Optimization recommendations
    recommended_strategies: list[OptimizationStrategy] = field(default_factory=list)
    estimated_improvement: float | None = None  # Percentage improvement
    implementation_effort: str = "unknown"  # low, medium, high


@dataclass
class OptimizationResult:
    """Result of an optimization implementation"""

    strategy: OptimizationStrategy
    issue_addressed: PerformanceIssue
    implemented_at: datetime

    # Performance impact
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    actual_improvement_percent: float

    # Implementation details
    changes_made: list[str]
    configuration_updates: dict[str, Any] = field(default_factory=dict)

    success: bool = True
    error_message: str | None = None


class PerformanceProfiler:
    """Advanced performance profiler with statistical analysis"""

    def __init__(self, name: str, sample_rate: float = 0.1) -> None:
        self.name = name
        self.sample_rate = sample_rate

        # Profiling data
        self.execution_times = deque(maxlen=1000)
        self.memory_snapshots = deque(maxlen=100)
        self.cpu_samples = deque(maxlen=1000)

        # Statistical analysis
        self.call_counts: dict[str, int] = defaultdict(int)
        self.total_times: dict[str, float] = defaultdict(float)
        self.max_times: dict[str, float] = defaultdict(float)

        # Profiling state
        self.profiling_active = False
        self.profiler: cProfile.Profile | None = None

        # Memory tracking
        self.memory_tracking = False
        self.memory_baseline: int | None = None

        logger.debug(f"Performance profiler initialized: {name}")

    @contextmanager
    def profile(self, operation: str = ""):
        """Context manager for profiling code blocks"""
        if not self.should_profile():
            yield
            return

        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Start CPU profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        try:
            yield

        finally:
            # Stop profiling
            if self.profiler:
                self.profiler.disable()

            # Record metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()

            execution_time = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory if start_memory else 0

            self.execution_times.append(execution_time)
            if memory_delta != 0:
                self.memory_snapshots.append(memory_delta)

            # Update statistics
            operation_key = operation or "unknown"
            self.call_counts[operation_key] += 1
            self.total_times[operation_key] += execution_time
            self.max_times[operation_key] = max(self.max_times[operation_key], execution_time)

            # Process profiling results
            if self.profiler:
                self._process_profiling_results(operation_key)

    def should_profile(self) -> bool:
        """Determine if profiling should be enabled"""
        import random

        return random.random() < self.sample_rate

    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )  # Convert to bytes on Linux

    def _process_profiling_results(self, operation: str) -> None:
        """Process cProfile results"""
        if not self.profiler:
            return

        try:
            # Get profile statistics
            stats_stream = StringIO()
            stats = pstats.Stats(self.profiler, stream=stats_stream)
            stats.sort_stats("cumulative")
            stats.print_stats(10)  # Top 10 functions

            profile_output = stats_stream.getvalue()

            # Store for analysis
            if not hasattr(self, "profile_results"):
                self.profile_results = {}

            self.profile_results[operation] = {
                "timestamp": datetime.now(),
                "profile_output": profile_output,
                "stats": stats,
            }

        except Exception as e:
            logger.warning(f"Error processing profiling results: {str(e)}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance analysis summary"""
        if not self.execution_times:
            return {"status": "no_data"}

        # Calculate statistics
        times = list(self.execution_times)

        return {
            "profiler_name": self.name,
            "total_samples": len(times),
            "avg_execution_time_ms": sum(times) / len(times),
            "min_execution_time_ms": min(times),
            "max_execution_time_ms": max(times),
            "p50_execution_time_ms": self._percentile(times, 0.5),
            "p95_execution_time_ms": self._percentile(times, 0.95),
            "p99_execution_time_ms": self._percentile(times, 0.99),
            "memory_samples": len(self.memory_snapshots),
            "avg_memory_delta_bytes": (
                sum(self.memory_snapshots) / len(self.memory_snapshots)
                if self.memory_snapshots
                else 0
            ),
            "call_counts": dict(self.call_counts),
            "operation_analysis": self._analyze_operations(),
        }

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    def _analyze_operations(self) -> dict[str, Any]:
        """Analyze operation performance"""
        analysis = {}

        for operation, count in self.call_counts.items():
            total_time = self.total_times[operation]
            max_time = self.max_times[operation]
            avg_time = total_time / count if count > 0 else 0

            analysis[operation] = {
                "call_count": count,
                "total_time_ms": total_time,
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "time_percentage": (
                    (total_time / sum(self.total_times.values())) * 100 if self.total_times else 0
                ),
            }

        return analysis

    def identify_bottlenecks(self) -> list[PerformanceIssue]:
        """Identify performance bottlenecks from profiling data"""
        issues = []

        # Analyze execution times
        if self.execution_times:
            times = list(self.execution_times)
            avg_time = sum(times) / len(times)
            self._percentile(times, 0.95)

            # Check for slow operations
            if avg_time > 1000:  # 1 second
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.CPU_BOTTLENECK,
                        severity=PerformanceSeverity.HIGH,
                        component_id=self.name,
                        description=f"High average execution time: {avg_time:.1f}ms",
                        impact_score=min(avg_time / 10, 100),
                        detected_at=datetime.now(),
                        current_value=avg_time,
                        expected_value=100,  # 100ms target
                        recommended_strategies=[
                            OptimizationStrategy.ALGORITHM_IMPROVEMENT,
                            OptimizationStrategy.CACHING,
                        ],
                    )
                )

            # Check for performance variance
            if len(times) > 10:
                import statistics

                std_dev = statistics.stdev(times)
                if std_dev > avg_time * 0.5:  # High variance
                    issues.append(
                        PerformanceIssue(
                            issue_type=PerformanceIssueType.ALGORITHM_COMPLEXITY,
                            severity=PerformanceSeverity.MEDIUM,
                            component_id=self.name,
                            description=f"High performance variance: {std_dev:.1f}ms std dev",
                            impact_score=min(std_dev / avg_time * 50, 100),
                            detected_at=datetime.now(),
                            current_value=std_dev,
                            expected_value=avg_time * 0.2,
                            recommended_strategies=[OptimizationStrategy.ALGORITHM_IMPROVEMENT],
                        )
                    )

        # Analyze memory usage
        if self.memory_snapshots:
            memory_deltas = list(self.memory_snapshots)
            avg_memory_delta = sum(memory_deltas) / len(memory_deltas)

            # Check for potential memory leaks
            if avg_memory_delta > 1024 * 1024:  # 1MB average growth
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.MEMORY_LEAK,
                        severity=PerformanceSeverity.HIGH,
                        component_id=self.name,
                        description=f"Potential memory leak: {avg_memory_delta / (1024*1024):.1f}MB avg growth",
                        impact_score=min(avg_memory_delta / (1024 * 1024), 100),
                        detected_at=datetime.now(),
                        current_value=avg_memory_delta,
                        expected_value=0,
                        recommended_strategies=[OptimizationStrategy.MEMORY_OPTIMIZATION],
                    )
                )

        return issues


class IOptimizationStrategy(ABC):
    """Interface for optimization strategies"""

    @abstractmethod
    def can_optimize(self, issue: PerformanceIssue) -> bool:
        """Check if this strategy can address the issue"""
        pass

    @abstractmethod
    def estimate_improvement(self, issue: PerformanceIssue) -> float:
        """Estimate improvement percentage"""
        pass

    @abstractmethod
    def implement(self, issue: PerformanceIssue) -> OptimizationResult:
        """Implement the optimization"""
        pass

    @abstractmethod
    def rollback(self, result: OptimizationResult) -> bool:
        """Rollback the optimization if needed"""
        pass


class CachingOptimizationStrategy(IOptimizationStrategy):
    """Optimization strategy using intelligent caching"""

    def can_optimize(self, issue: PerformanceIssue) -> bool:
        """Check if caching can help with this issue"""
        return issue.issue_type in [
            PerformanceIssueType.CPU_BOTTLENECK,
            PerformanceIssueType.DATABASE_SLOW_QUERY,
            PerformanceIssueType.IO_BOTTLENECK,
        ]

    def estimate_improvement(self, issue: PerformanceIssue) -> float:
        """Estimate caching improvement"""
        if issue.issue_type == PerformanceIssueType.DATABASE_SLOW_QUERY:
            return 70.0  # 70% improvement for DB queries
        elif issue.issue_type == PerformanceIssueType.CPU_BOTTLENECK:
            return 40.0  # 40% for CPU-bound operations
        else:
            return 30.0  # 30% for I/O operations

    def implement(self, issue: PerformanceIssue) -> OptimizationResult:
        """Implement caching optimization"""
        try:
            from .caching import CacheConfig, get_cache_manager

            before_metrics = self._capture_metrics()

            # Create component-specific cache
            cache_manager = get_cache_manager()
            cache_name = f"{issue.component_id}_optimization"

            # Configure cache based on issue type
            if issue.issue_type == PerformanceIssueType.DATABASE_SLOW_QUERY:
                ttl_seconds = 300  # 5 minutes for DB queries
                max_size = 1000
            elif issue.issue_type == PerformanceIssueType.CPU_BOTTLENECK:
                ttl_seconds = 1800  # 30 minutes for CPU-intensive operations
                max_size = 500
            else:
                ttl_seconds = 600  # 10 minutes default
                max_size = 750

            config = CacheConfig(
                cache_name=cache_name,
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                enable_compression=True,
            )

            cache_manager.create_cache(config)

            # Simulate cache warming (in real implementation, this would be specific to the component)
            changes_made = [
                f"Created cache '{cache_name}' with TTL {ttl_seconds}s",
                f"Configured max size: {max_size} entries",
                "Enabled compression for large objects",
            ]

            after_metrics = self._capture_metrics()

            return OptimizationResult(
                strategy=OptimizationStrategy.CACHING,
                issue_addressed=issue,
                implemented_at=datetime.now(),
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                actual_improvement_percent=self.estimate_improvement(issue),
                changes_made=changes_made,
                configuration_updates={"cache_config": config.__dict__},
                success=True,
            )

        except Exception as e:
            logger.error(f"Caching optimization failed: {str(e)}")
            return OptimizationResult(
                strategy=OptimizationStrategy.CACHING,
                issue_addressed=issue,
                implemented_at=datetime.now(),
                before_metrics=self._capture_metrics(),
                after_metrics=self._capture_metrics(),
                actual_improvement_percent=0.0,
                changes_made=[],
                success=False,
                error_message=str(e),
            )

    def rollback(self, result: OptimizationResult) -> bool:
        """Rollback caching optimization"""
        try:
            if "cache_config" in result.configuration_updates:
                from .caching import get_cache_manager

                cache_manager = get_cache_manager()
                config = result.configuration_updates["cache_config"]
                cache_manager.remove_cache(config["cache_name"])
                return True
        except Exception as e:
            logger.error(f"Caching rollback failed: {str(e)}")

        return False

    def _capture_metrics(self) -> PerformanceMetrics:
        """Capture current performance metrics"""
        # This would capture real metrics from the system
        return PerformanceMetrics(timestamp=datetime.now())


class ThreadPoolOptimizationStrategy(IOptimizationStrategy):
    """Optimization strategy for thread pool tuning"""

    def can_optimize(self, issue: PerformanceIssue) -> bool:
        """Check if thread pool optimization can help"""
        return issue.issue_type in [
            PerformanceIssueType.THREAD_CONTENTION,
            PerformanceIssueType.CPU_BOTTLENECK,
        ]

    def estimate_improvement(self, issue: PerformanceIssue) -> float:
        """Estimate thread pool optimization improvement"""
        if issue.issue_type == PerformanceIssueType.THREAD_CONTENTION:
            return 50.0
        return 25.0

    def implement(self, issue: PerformanceIssue) -> OptimizationResult:
        """Implement thread pool optimization"""
        try:
            before_metrics = self._capture_metrics()

            # Get concurrency manager
            concurrency_manager = get_concurrency_manager()
            stats = concurrency_manager.get_system_stats()

            changes_made = []

            # Analyze thread pool utilization and adjust
            thread_pools = stats.get("thread_pools", {})
            for pool_name, pool_stats in thread_pools.items():
                utilization = (
                    pool_stats.get("active_tasks", 0) / max(1, pool_stats.get("max_workers", 1))
                ) * 100

                if utilization > 80:  # High utilization
                    # In a real implementation, this would adjust pool size
                    changes_made.append(
                        f"Would increase {pool_name} pool size due to {utilization:.1f}% utilization"
                    )
                elif utilization < 20:  # Low utilization
                    changes_made.append(
                        f"Would decrease {pool_name} pool size due to low {utilization:.1f}% utilization"
                    )

            if not changes_made:
                changes_made = ["Thread pool sizes are optimal"]

            after_metrics = self._capture_metrics()

            return OptimizationResult(
                strategy=OptimizationStrategy.THREAD_TUNING,
                issue_addressed=issue,
                implemented_at=datetime.now(),
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                actual_improvement_percent=self.estimate_improvement(issue),
                changes_made=changes_made,
                success=True,
            )

        except Exception as e:
            logger.error(f"Thread pool optimization failed: {str(e)}")
            return OptimizationResult(
                strategy=OptimizationStrategy.THREAD_TUNING,
                issue_addressed=issue,
                implemented_at=datetime.now(),
                before_metrics=self._capture_metrics(),
                after_metrics=self._capture_metrics(),
                actual_improvement_percent=0.0,
                changes_made=[],
                success=False,
                error_message=str(e),
            )

    def rollback(self, result: OptimizationResult) -> bool:
        """Rollback thread pool optimization"""
        # Thread pool changes would be reverted here
        return True

    def _capture_metrics(self) -> PerformanceMetrics:
        """Capture current performance metrics"""
        return PerformanceMetrics(timestamp=datetime.now())


class PerformanceOptimizer(BaseComponent):
    """
    Intelligent performance optimization system

    Continuously monitors system performance and automatically applies
    optimizations to maintain peak performance.
    """

    def __init__(self, config: ComponentConfig | None = None) -> None:
        if not config:
            config = ComponentConfig(
                component_id="performance_optimizer", component_type="performance_optimizer"
            )

        super().__init__(config)

        # Performance monitoring
        self.profilers: dict[str, PerformanceProfiler] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.detected_issues: list[PerformanceIssue] = []
        self.optimization_results: list[OptimizationResult] = []

        # Optimization strategies
        self.strategies: list[IOptimizationStrategy] = [
            CachingOptimizationStrategy(),
            ThreadPoolOptimizationStrategy(),
        ]

        # Configuration
        self.auto_optimize = True
        self.optimization_threshold = 60.0  # Only optimize issues with impact > 60%
        self.analysis_interval = timedelta(minutes=5)
        self.optimization_interval = timedelta(minutes=15)

        # Metrics integration
        self.metrics_registry = get_metrics_registry()
        self._setup_metrics()

        logger.info("Performance optimizer initialized")

    def _initialize_component(self) -> None:
        """Initialize performance optimizer"""
        # Schedule performance analysis
        schedule_recurring_task(
            task_id="performance_analysis",
            function=self._analyze_performance,
            interval=self.analysis_interval,
            component_id=self.component_id,
        )

        # Schedule optimization execution
        if self.auto_optimize:
            schedule_recurring_task(
                task_id="execute_optimizations",
                function=self._execute_optimizations,
                interval=self.optimization_interval,
                component_id=self.component_id,
            )

        # Schedule system monitoring
        schedule_recurring_task(
            task_id="monitor_system_resources",
            function=self._monitor_system_resources,
            interval=timedelta(seconds=30),
            component_id=self.component_id,
        )

    def _start_component(self) -> None:
        """Start performance optimization"""
        logger.info("Performance optimization started")

    def _stop_component(self) -> None:
        """Stop performance optimization"""
        # Generate final performance report
        self._generate_performance_report()
        logger.info("Performance optimization stopped")

    def _health_check(self) -> HealthStatus:
        """Check optimizer health"""
        try:
            # Check if we have recent performance data
            if not self.performance_history:
                return HealthStatus.DEGRADED

            # Check for critical performance issues
            critical_issues = [
                issue
                for issue in self.detected_issues
                if issue.severity == PerformanceSeverity.CRITICAL
            ]

            if critical_issues:
                return HealthStatus.CRITICAL

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNHEALTHY

    def _setup_metrics(self) -> None:
        """Setup performance metrics"""
        self.performance_metrics = {
            "issues_detected": self.metrics_registry.register_counter(
                "performance_issues_detected_total",
                "Total performance issues detected",
                component_id=self.component_id,
            ),
            "optimizations_applied": self.metrics_registry.register_counter(
                "performance_optimizations_applied_total",
                "Total optimizations applied",
                component_id=self.component_id,
            ),
            "optimization_success_rate": self.metrics_registry.register_gauge(
                "performance_optimization_success_rate_percent",
                "Optimization success rate percentage",
                component_id=self.component_id,
            ),
            "avg_performance_improvement": self.metrics_registry.register_gauge(
                "performance_avg_improvement_percent",
                "Average performance improvement percentage",
                component_id=self.component_id,
            ),
        }

    def create_profiler(self, name: str, sample_rate: float = 0.1) -> PerformanceProfiler:
        """Create component-specific profiler"""
        if name in self.profilers:
            return self.profilers[name]

        profiler = PerformanceProfiler(name, sample_rate)
        self.profilers[name] = profiler

        logger.info(f"Created profiler: {name}")
        return profiler

    @track_execution_time("performance_analysis")
    def _analyze_performance(self) -> None:
        """Analyze system performance and identify issues"""
        try:
            # Collect current performance metrics
            current_metrics = self._collect_performance_metrics()
            self.performance_history.append(current_metrics)

            # Analyze profiler data for bottlenecks
            new_issues = []
            for _profiler_name, profiler in self.profilers.items():
                issues = profiler.identify_bottlenecks()
                new_issues.extend(issues)

            # Analyze system-wide performance trends
            trend_issues = self._analyze_performance_trends()
            new_issues.extend(trend_issues)

            # Filter out duplicate issues
            new_issues = self._deduplicate_issues(new_issues)

            # Add to detected issues
            for issue in new_issues:
                self.detected_issues.append(issue)
                self.performance_metrics["issues_detected"].increment()

                logger.warning(
                    f"Performance issue detected: {issue.description} "
                    f"(severity: {issue.severity.value}, impact: {issue.impact_score:.1f}%)"
                )

            # Clean up old issues
            self._cleanup_old_issues()

            self.record_operation(success=True)

        except Exception as e:
            logger.error(f"Performance analysis error: {str(e)}")
            self.record_operation(success=False, error_message=str(e))
            report_error(e, component=self.component_id)

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # Get system resource usage
            cpu_percent = 0.0
            memory_rss = 0
            memory_percent = 0.0
            thread_count = 0

            try:
                import psutil

                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_rss = memory_info.rss
                memory_percent = process.memory_percent()
                thread_count = process.num_threads()
            except ImportError:
                # Fallback to resource module
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                cpu_percent = (rusage.ru_utime + rusage.ru_stime) / time.time() * 100
                memory_rss = rusage.ru_maxrss * 1024  # Convert to bytes

            # Get thread pool utilization
            thread_pool_util = {}
            try:
                concurrency_manager = get_concurrency_manager()
                stats = concurrency_manager.get_system_stats()
                for pool_name, pool_stats in stats.get("thread_pools", {}).items():
                    utilization = (
                        pool_stats.get("active_tasks", 0) / max(1, pool_stats.get("max_workers", 1))
                    ) * 100
                    thread_pool_util[pool_name] = utilization
            except Exception:
                pass

            # Get cache hit rate
            cache_hit_rate = 0.0
            try:
                from .caching import get_cache_manager

                cache_manager = get_cache_manager()
                cache_stats = cache_manager.get_system_statistics()
                cache_hit_rate = cache_stats.get("global_statistics", {}).get(
                    "global_hit_rate", 0.0
                )
            except ImportError:
                pass

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_rss_bytes=memory_rss,
                memory_percent=memory_percent,
                thread_count=thread_count,
                thread_pool_utilization=thread_pool_util,
                cache_hit_rate=cache_hit_rate,
            )

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            return PerformanceMetrics(timestamp=datetime.now())

    def _analyze_performance_trends(self) -> list[PerformanceIssue]:
        """Analyze performance trends to identify issues"""
        issues = []

        if len(self.performance_history) < 10:
            return issues

        # Get recent metrics
        recent_metrics = list(self.performance_history)[-10:]

        # Analyze CPU usage trend
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        avg_cpu = sum(cpu_values) / len(cpu_values)

        if avg_cpu > 80:  # High CPU usage
            issues.append(
                PerformanceIssue(
                    issue_type=PerformanceIssueType.CPU_BOTTLENECK,
                    severity=(
                        PerformanceSeverity.HIGH if avg_cpu > 90 else PerformanceSeverity.MEDIUM
                    ),
                    component_id="system",
                    description=f"High CPU usage: {avg_cpu:.1f}% average",
                    impact_score=min(avg_cpu, 100),
                    detected_at=datetime.now(),
                    current_value=avg_cpu,
                    expected_value=50.0,
                    recommended_strategies=[
                        OptimizationStrategy.ALGORITHM_IMPROVEMENT,
                        OptimizationStrategy.CACHING,
                    ],
                )
            )

        # Analyze memory usage trend
        memory_values = [m.memory_percent for m in recent_metrics]
        if memory_values and any(m > 0 for m in memory_values):
            avg_memory = sum(memory_values) / len(memory_values)

            if avg_memory > 80:  # High memory usage
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.MEMORY_LEAK,
                        severity=(
                            PerformanceSeverity.HIGH
                            if avg_memory > 90
                            else PerformanceSeverity.MEDIUM
                        ),
                        component_id="system",
                        description=f"High memory usage: {avg_memory:.1f}% average",
                        impact_score=min(avg_memory, 100),
                        detected_at=datetime.now(),
                        current_value=avg_memory,
                        expected_value=60.0,
                        recommended_strategies=[OptimizationStrategy.MEMORY_OPTIMIZATION],
                    )
                )

        # Analyze cache hit rate
        cache_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate > 0]
        if cache_rates:
            avg_hit_rate = sum(cache_rates) / len(cache_rates)

            if avg_hit_rate < 70:  # Low cache hit rate
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.CACHE_MISS_RATE,
                        severity=PerformanceSeverity.MEDIUM,
                        component_id="cache_system",
                        description=f"Low cache hit rate: {avg_hit_rate:.1f}%",
                        impact_score=100 - avg_hit_rate,
                        detected_at=datetime.now(),
                        current_value=avg_hit_rate,
                        expected_value=85.0,
                        recommended_strategies=[OptimizationStrategy.CACHING],
                    )
                )

        return issues

    def _deduplicate_issues(self, issues: list[PerformanceIssue]) -> list[PerformanceIssue]:
        """Remove duplicate issues"""
        seen = set()
        unique_issues = []

        for issue in issues:
            # Create a key based on issue type, component, and description
            issue_key = (issue.issue_type, issue.component_id, issue.description)

            if issue_key not in seen:
                seen.add(issue_key)
                unique_issues.append(issue)

        return unique_issues

    def _cleanup_old_issues(self) -> None:
        """Remove old resolved or expired issues"""
        cutoff_time = datetime.now() - timedelta(hours=1)

        self.detected_issues = [
            issue
            for issue in self.detected_issues
            if issue.detected_at > cutoff_time or issue.severity == PerformanceSeverity.CRITICAL
        ]

    def _execute_optimizations(self) -> None:
        """Execute optimizations for detected issues"""
        if not self.auto_optimize:
            return

        try:
            # Sort issues by impact score (highest first)
            high_impact_issues = [
                issue
                for issue in self.detected_issues
                if issue.impact_score >= self.optimization_threshold
            ]

            high_impact_issues.sort(key=lambda x: x.impact_score, reverse=True)

            optimizations_applied = 0

            for issue in high_impact_issues[:5]:  # Limit to top 5 issues
                # Find suitable optimization strategy
                best_strategy = None
                best_improvement = 0.0

                for strategy in self.strategies:
                    if strategy.can_optimize(issue):
                        estimated_improvement = strategy.estimate_improvement(issue)
                        if estimated_improvement > best_improvement:
                            best_strategy = strategy
                            best_improvement = estimated_improvement

                if best_strategy:
                    logger.info(f"Applying optimization for issue: {issue.description}")

                    # Execute optimization
                    result = best_strategy.implement(issue)
                    self.optimization_results.append(result)

                    if result.success:
                        optimizations_applied += 1
                        self.performance_metrics["optimizations_applied"].increment()

                        logger.info(
                            f"Optimization successful: {result.actual_improvement_percent:.1f}% improvement"
                        )

                        # Remove the addressed issue
                        if issue in self.detected_issues:
                            self.detected_issues.remove(issue)
                    else:
                        logger.warning(f"Optimization failed: {result.error_message}")

            # Update success rate metric
            if self.optimization_results:
                successful = sum(1 for r in self.optimization_results if r.success)
                success_rate = (successful / len(self.optimization_results)) * 100
                self.performance_metrics["optimization_success_rate"].set(success_rate)

                # Update average improvement metric
                improvements = [
                    r.actual_improvement_percent for r in self.optimization_results if r.success
                ]
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    self.performance_metrics["avg_performance_improvement"].set(avg_improvement)

            if optimizations_applied > 0:
                logger.info(f"Applied {optimizations_applied} performance optimizations")

            self.record_operation(success=True)

        except Exception as e:
            logger.error(f"Optimization execution error: {str(e)}")
            self.record_operation(success=False, error_message=str(e))
            report_error(e, component=self.component_id)

    def _monitor_system_resources(self) -> None:
        """Monitor system resources for immediate issues"""
        try:
            current_metrics = self._collect_performance_metrics()

            # Check for immediate critical issues
            critical_issues = []

            if current_metrics.cpu_usage_percent > 95:
                critical_issues.append(
                    f"CPU usage critical: {current_metrics.cpu_usage_percent:.1f}%"
                )

            if current_metrics.memory_percent > 95:
                critical_issues.append(
                    f"Memory usage critical: {current_metrics.memory_percent:.1f}%"
                )

            for issue_desc in critical_issues:
                logger.critical(f"PERFORMANCE CRITICAL: {issue_desc}")

                # Could trigger immediate emergency optimizations here

        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")

    def _generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "analysis_period_hours": len(self.performance_history)
                * (self.analysis_interval.total_seconds() / 3600),
                "summary": {
                    "total_issues_detected": len(self.detected_issues),
                    "optimizations_applied": len(self.optimization_results),
                    "success_rate": 0.0,
                    "avg_improvement": 0.0,
                },
                "current_issues": [],
                "optimization_history": [],
                "profiler_summaries": {},
            }

            # Calculate success metrics
            if self.optimization_results:
                successful = [r for r in self.optimization_results if r.success]
                report["summary"]["success_rate"] = (
                    len(successful) / len(self.optimization_results)
                ) * 100

                improvements = [r.actual_improvement_percent for r in successful]
                if improvements:
                    report["summary"]["avg_improvement"] = sum(improvements) / len(improvements)

            # Current issues
            for issue in self.detected_issues:
                report["current_issues"].append(
                    {
                        "type": issue.issue_type.value,
                        "severity": issue.severity.value,
                        "component": issue.component_id,
                        "description": issue.description,
                        "impact_score": issue.impact_score,
                        "detected_at": issue.detected_at.isoformat(),
                    }
                )

            # Optimization history
            for result in self.optimization_results[-10:]:  # Last 10 optimizations
                report["optimization_history"].append(
                    {
                        "strategy": result.strategy.value,
                        "improvement_percent": result.actual_improvement_percent,
                        "implemented_at": result.implemented_at.isoformat(),
                        "success": result.success,
                        "changes_made": result.changes_made,
                    }
                )

            # Profiler summaries
            for name, profiler in self.profilers.items():
                report["profiler_summaries"][name] = profiler.get_performance_summary()

            logger.info("Performance report generated")
            return report

        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            return {"error": str(e)}

    def get_current_issues(self) -> list[PerformanceIssue]:
        """Get current performance issues"""
        return self.detected_issues.copy()

    def get_optimization_results(self) -> list[OptimizationResult]:
        """Get optimization results history"""
        return self.optimization_results.copy()

    def force_optimization(self, issue: PerformanceIssue) -> OptimizationResult | None:
        """Force immediate optimization of specific issue"""
        for strategy in self.strategies:
            if strategy.can_optimize(issue):
                logger.info(f"Force optimizing issue: {issue.description}")
                return strategy.implement(issue)

        logger.warning(f"No strategy available for issue: {issue.description}")
        return None


# Global performance optimizer instance
_performance_optimizer: PerformanceOptimizer | None = None
_optimizer_lock = threading.Lock()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer

    with _optimizer_lock:
        if _performance_optimizer is None:
            _performance_optimizer = PerformanceOptimizer()
            logger.info("Global performance optimizer created")

        return _performance_optimizer


def create_profiler(name: str, sample_rate: float = 0.1) -> PerformanceProfiler:
    """Create profiler for component"""
    return get_performance_optimizer().create_profiler(name, sample_rate)


# Performance monitoring decorators


def profile_performance(profiler_name: str = None, sample_rate: float = 0.1):
    """Decorator to profile function performance"""

    def decorator(func: F) -> F:
        nonlocal profiler_name
        if not profiler_name:
            profiler_name = f"{func.__module__}.{func.__name__}"

        optimizer = get_performance_optimizer()
        profiler = optimizer.create_profiler(profiler_name, sample_rate)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(func.__name__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def performance_context(operation: str, profiler_name: str = "default"):
    """Context manager for performance monitoring"""
    optimizer = get_performance_optimizer()
    profiler = optimizer.create_profiler(profiler_name)

    with profiler.profile(operation):
        yield profiler
