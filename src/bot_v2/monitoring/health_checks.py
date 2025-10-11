"""Backwards-compatible shim for health monitoring utilities."""

from __future__ import annotations

from bot_v2.utilities.performance_monitoring import get_performance_health_check

from .health import (
    APIHealthCheck,
    BrokerageHealthCheck,
    DatabaseHealthCheck,
    HealthCheckEndpoint,
    HealthChecker,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus,
    MemoryHealthCheck,
    PerformanceHealthCheck,
    get_health_registry,
    get_health_summary,
    setup_basic_health_checks,
)
from .health import checks as _health_checks

psutil = _health_checks.psutil


__all__ = [
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",
    "DatabaseHealthCheck",
    "APIHealthCheck",
    "BrokerageHealthCheck",
    "MemoryHealthCheck",
    "PerformanceHealthCheck",
    "HealthCheckRegistry",
    "get_health_registry",
    "HealthCheckEndpoint",
    "setup_basic_health_checks",
    "get_health_summary",
    "psutil",
    "get_performance_health_check",
]
