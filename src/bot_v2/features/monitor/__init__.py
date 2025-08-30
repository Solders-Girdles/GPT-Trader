"""
System monitoring feature slice - health, performance, and alerting.

Complete isolation - no external dependencies.
"""

from .monitor import start_monitoring, stop_monitoring, get_system_health, get_alerts
from .types import SystemHealth, Alert, PerformanceMetrics, ResourceUsage

__all__ = [
    'start_monitoring',
    'stop_monitoring',
    'get_system_health',
    'get_alerts',
    'SystemHealth',
    'Alert',
    'PerformanceMetrics',
    'ResourceUsage'
]