"""
Performance Monitor for Bot V2 Orchestration System

Tracks execution times, method call counts, and performance metrics
for orchestration components with thread-safe operations.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Stores performance metrics for a specific operation."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        return self.total_time / self.count if self.count > 0 else 0.0


class PerformanceMonitor:
    """
    Thread-safe performance monitoring for orchestration operations.
    
    Tracks execution times and provides statistical analysis of
    slice performance and method call patterns.
    """
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        self._operation_history: List[tuple] = []
        self._max_history = 1000
    
    @contextmanager
    def track(self, operation: str):
        """
        Context manager to track operation execution time.
        
        Args:
            operation: Name of the operation being tracked
            
        Usage:
            with monitor.track("slice.backtest.run"):
                # operation code here
                pass
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._record(operation, elapsed)
    
    def _record(self, operation: str, elapsed: float):
        """Record performance metrics for an operation."""
        with self._lock:
            # Update metrics
            if operation not in self._metrics:
                self._metrics[operation] = PerformanceMetrics()
            
            metrics = self._metrics[operation]
            metrics.count += 1
            metrics.total_time += elapsed
            metrics.min_time = min(metrics.min_time, elapsed)
            metrics.max_time = max(metrics.max_time, elapsed)
            
            # Track history (keep recent operations)
            self._operation_history.append((operation, elapsed, time.time()))
            if len(self._operation_history) > self._max_history:
                self._operation_history.pop(0)
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            operation: Specific operation to get metrics for, or None for all
            
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            if operation:
                return self._format_metrics(self._metrics.get(operation))
            return {op: self._format_metrics(m) for op, m in self._metrics.items()}
    
    def _format_metrics(self, metrics: Optional[PerformanceMetrics]) -> Dict[str, Any]:
        """Format metrics for output."""
        if not metrics or metrics.count == 0:
            return {
                'count': 0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_time': 0.0
            }
        
        return {
            'count': metrics.count,
            'avg_time': round(metrics.avg_time, 4),
            'min_time': round(metrics.min_time, 4),
            'max_time': round(metrics.max_time, 4),
            'total_time': round(metrics.total_time, 4)
        }
    
    def get_slowest_operations(self, limit: int = 5) -> List[tuple]:
        """
        Get the slowest operations by average time.
        
        Args:
            limit: Number of operations to return
            
        Returns:
            List of tuples (operation_name, avg_time)
        """
        with self._lock:
            sorted_ops = sorted(
                [(op, metrics.avg_time) for op, metrics in self._metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_ops[:limit]
    
    def get_most_frequent_operations(self, limit: int = 5) -> List[tuple]:
        """
        Get the most frequently called operations.
        
        Args:
            limit: Number of operations to return
            
        Returns:
            List of tuples (operation_name, call_count)
        """
        with self._lock:
            sorted_ops = sorted(
                [(op, metrics.count) for op, metrics in self._metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_ops[:limit]
    
    def reset_metrics(self, operation: Optional[str] = None):
        """
        Reset performance metrics.
        
        Args:
            operation: Specific operation to reset, or None to reset all
        """
        with self._lock:
            if operation:
                if operation in self._metrics:
                    del self._metrics[operation]
            else:
                self._metrics.clear()
                self._operation_history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        with self._lock:
            if not self._metrics:
                return {
                    'total_operations': 0,
                    'total_calls': 0,
                    'slowest_operations': [],
                    'most_frequent_operations': []
                }
            
            total_calls = sum(m.count for m in self._metrics.values())
            
            return {
                'total_operations': len(self._metrics),
                'total_calls': total_calls,
                'slowest_operations': self.get_slowest_operations(),
                'most_frequent_operations': self.get_most_frequent_operations()
            }


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def track_performance(operation: str):
    """
    Decorator to track performance of a function.
    
    Args:
        operation: Name of the operation being tracked
        
    Usage:
        @track_performance("slice.backtest.analyze")
        def analyze_data():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _performance_monitor.track(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator