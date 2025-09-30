"""System-level monitoring interfaces and helpers."""

from bot_v2.monitoring.interfaces import Alert, PerformanceMetrics, ResourceUsage, SystemHealth
from bot_v2.monitoring.system.engine import (
    MonitoringSystem,
    acknowledge_alert,
    get_active_alerts,
    get_alerts,
    get_system_health,
    resolve_alert,
    start_monitoring,
    stop_monitoring,
)
from bot_v2.monitoring.system.logger import (
    LogLevel,
    ProductionLogger,
    get_correlation_id,
    get_logger,
    log_error,
    log_event,
    log_ml_prediction,
    log_performance,
    log_trade,
    set_correlation_id,
)

__all__ = [
    "start_monitoring",
    "stop_monitoring",
    "get_system_health",
    "get_alerts",
    "get_active_alerts",
    "MonitoringSystem",
    "Alert",
    "SystemHealth",
    "PerformanceMetrics",
    "ResourceUsage",
    "get_logger",
    "log_event",
    "log_trade",
    "log_ml_prediction",
    "log_performance",
    "log_error",
    "set_correlation_id",
    "get_correlation_id",
    "LogLevel",
    "ProductionLogger",
    "acknowledge_alert",
    "resolve_alert",
]
