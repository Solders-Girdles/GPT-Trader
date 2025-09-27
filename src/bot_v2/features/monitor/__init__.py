"""
System monitoring feature slice - health, performance, and alerting.

Complete isolation - no external dependencies.
"""

from .monitor import start_monitoring, stop_monitoring, get_system_health, get_alerts
from .types import SystemHealth, Alert, PerformanceMetrics, ResourceUsage
from .logger import (
    get_logger, log_event, log_trade, log_ml_prediction, log_performance, log_error,
    set_correlation_id, get_correlation_id, LogLevel, ProductionLogger
)

__all__ = [
    'start_monitoring',
    'stop_monitoring',
    'get_system_health',
    'get_alerts',
    'SystemHealth',
    'Alert',
    'PerformanceMetrics',
    'ResourceUsage',
    # Logging exports
    'get_logger',
    'log_event',
    'log_trade',
    'log_ml_prediction', 
    'log_performance',
    'log_error',
    'set_correlation_id',
    'get_correlation_id',
    'LogLevel',
    'ProductionLogger'
]
