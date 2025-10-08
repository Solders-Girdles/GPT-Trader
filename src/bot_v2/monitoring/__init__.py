"""
Monitoring module for the GPT-Trader bot system.

This module provides comprehensive monitoring capabilities including:
- Health checks for system components
- Performance metrics collection
- Alerting and notification systems
- Runtime guards and configuration monitoring
- System telemetry and observability

Key Components:
- Health checks: Modular health check system for components
- Metrics: Performance and business metrics collection
- Alerts: Configurable alerting system
- Guards: Runtime protection and monitoring
- Telemetry: System observability and tracking
"""

from .alert_types import Alert, AlertSeverity
from .alerting_system import AlertingSystem, AlertRule
from .configuration_guardian import ConfigurationGuardian
from .health_checks import (
    HealthCheckEndpoint,
    HealthChecker,
    HealthCheckResult,
    setup_basic_health_checks,
)
from .metrics_collector import MetricsCollector
from .monitoring_dashboard import (
    MonitoringDashboard,
    get_dashboard,
    record_metric,
    track_workflow,
    update_slice_status,
)
from .runtime_guards import RuntimeGuardManager
from .workflow_tracker import WorkflowExecution, WorkflowStatus, WorkflowTracker

__all__ = [
    # Alerts
    "Alert",
    "AlertSeverity",
    "AlertingSystem",
    "AlertRule",
    # Configuration
    "ConfigurationGuardian",
    # Health checks
    "HealthCheckResult",
    "HealthChecker",
    "HealthCheckEndpoint",
    "setup_basic_health_checks",
    # Metrics
    "MetricsCollector",
    # Monitoring dashboard
    "MonitoringDashboard",
    "get_dashboard",
    "record_metric",
    "track_workflow",
    "update_slice_status",
    # Runtime guards
    "RuntimeGuardManager",
    # Workflow tracking
    "WorkflowExecution",
    "WorkflowStatus",
    "WorkflowTracker",
]
