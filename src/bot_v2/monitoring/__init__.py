"""
Monitoring module for the GPT-Trader bot system.

This module exposes the production monitoring primitives that remain active
after the cleanup:
- Configuration drift detection (`ConfigurationGuardian`)
- Health checks for system components
- Runtime guard management for the execution loop
"""

from .alert_types import Alert, AlertSeverity
from .configuration_guardian import ConfigurationGuardian
from .health_checks import (
    HealthCheckEndpoint,
    HealthChecker,
    HealthCheckResult,
    setup_basic_health_checks,
)
from .runtime_guards import RuntimeGuardManager

__all__ = [
    "Alert",
    "AlertSeverity",
    "ConfigurationGuardian",
    "HealthCheckResult",
    "HealthChecker",
    "HealthCheckEndpoint",
    "setup_basic_health_checks",
    "RuntimeGuardManager",
]
