"""
Performance Monitor: Real-time metrics, alerts, and analytics.

Provides tools for:
- Real-time performance tracking
- Metrics aggregation and analysis
- Alert rules and notifications
- Regime-specific performance analytics
"""

from gpt_trader.features.strategy_dev.monitor.alerts import AlertManager, AlertRule
from gpt_trader.features.strategy_dev.monitor.metrics import (
    MetricsAggregator,
    PerformanceSnapshot,
)
from gpt_trader.features.strategy_dev.monitor.performance_monitor import PerformanceMonitor

__all__ = [
    "AlertManager",
    "AlertRule",
    "MetricsAggregator",
    "PerformanceMonitor",
    "PerformanceSnapshot",
]
