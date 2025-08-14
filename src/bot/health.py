"""
Health check and system monitoring utilities for GPT-Trader.

This module provides comprehensive health checking capabilities including:
- Component health monitoring
- System resource monitoring
- Dependency health checks
- Health status reporting
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

from .config import settings
from .performance import performance_monitor

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    timestamp: datetime
    checks: list[HealthCheck]
    summary: dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Centralized health checking system."""

    def __init__(self) -> None:
        self.checks: dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._last_check: SystemHealth | None = None

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        with self._lock:
            self.checks[name] = check_func

    def unregister_check(self, name: str) -> None:
        """Unregister a health check function."""
        with self._lock:
            self.checks.pop(name, None)

    async def run_check(self, name: str, check_func: Callable) -> HealthCheck:
        """Run a single health check."""
        start_time = datetime.now()
        start_perf = performance_monitor._process.memory_info().rss

        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            duration = (datetime.now() - start_time).total_seconds() * 1000
            end_perf = performance_monitor._process.memory_info().rss
            memory_delta = (end_perf - start_perf) / 1024 / 1024  # MB

            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "Check completed")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check completed" if result else "Check failed"
                details = {}

            details["memory_delta_mb"] = memory_delta

            return HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                details=details,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Health check '{name}' failed: {e}")

            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={"error": str(e)},
            )

    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        with self._lock:
            checks_to_run = list(self.checks.items())

        if not checks_to_run:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                checks=[],
                summary={"message": "No health checks registered"},
            )

        # Run all checks concurrently
        tasks = [self.run_check(name, func) for name, func in checks_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_checks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = checks_to_run[i][0]
                health_checks.append(
                    HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed with exception: {result}",
                        timestamp=datetime.now(),
                        duration_ms=0,
                        details={"error": str(result)},
                    )
                )
            else:
                health_checks.append(result)

        # Determine overall status
        status_counts = {}
        for check in health_checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1

        if HealthStatus.UNHEALTHY in status_counts:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in status_counts:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in status_counts:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        # Create summary
        summary = {
            "total_checks": len(health_checks),
            "status_counts": {status.value: count for status, count in status_counts.items()},
            "average_duration_ms": sum(check.duration_ms for check in health_checks)
            / len(health_checks),
            "checks_passed": len([c for c in health_checks if c.status == HealthStatus.HEALTHY]),
            "checks_failed": len([c for c in health_checks if c.status == HealthStatus.UNHEALTHY]),
            "checks_degraded": len([c for c in health_checks if c.status == HealthStatus.DEGRADED]),
        }

        system_health = SystemHealth(
            status=overall_status, timestamp=datetime.now(), checks=health_checks, summary=summary
        )

        self._last_check = system_health
        return system_health

    def get_last_check(self) -> SystemHealth | None:
        """Get the last health check result."""
        return self._last_check

    def is_healthy(self) -> bool:
        """Check if the system is healthy based on last check."""
        if not self._last_check:
            return False
        return self._last_check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# Global health checker instance
health_checker = HealthChecker()


# Built-in health checks
def check_system_resources() -> dict[str, Any]:
    """Check system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Define thresholds
        cpu_threshold = 90.0
        memory_threshold = 90.0
        disk_threshold = 90.0

        # Determine status
        if (
            cpu_percent > cpu_threshold
            or memory.percent > memory_threshold
            or disk.percent > disk_threshold
        ):
            status = HealthStatus.DEGRADED
            message = "System resources are high"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources are normal"

        return {
            "status": status.value,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            },
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"Failed to check system resources: {e}",
            "details": {"error": str(e)},
        }


def check_process_health() -> dict[str, Any]:
    """Check process health."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        # Define thresholds
        memory_threshold_mb = 2048  # 2GB
        cpu_threshold = 80.0

        memory_mb = memory_info.rss / 1024 / 1024

        if memory_mb > memory_threshold_mb or cpu_percent > cpu_threshold:
            status = HealthStatus.DEGRADED
            message = "Process resource usage is high"
        else:
            status = HealthStatus.HEALTHY
            message = "Process is healthy"

        return {
            "status": status.value,
            "message": message,
            "details": {
                "memory_usage_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
            },
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"Failed to check process health: {e}",
            "details": {"error": str(e)},
        }


def check_configuration() -> dict[str, Any]:
    """Check configuration health."""
    try:
        issues = []

        # Check Alpaca configuration
        if not settings.alpaca.api_key_id:
            issues.append("Alpaca API key not configured")

        # Check data directory
        if not settings.data.cache_dir.exists():
            issues.append("Data cache directory does not exist")

        # Check environment
        if settings.environment not in ["development", "testing", "production"]:
            issues.append(f"Invalid environment: {settings.environment}")

        if issues:
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": f"Configuration issues found: {', '.join(issues)}",
                "details": {"issues": issues},
            }
        else:
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Configuration is valid",
                "details": {"environment": settings.environment},
            }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"Failed to check configuration: {e}",
            "details": {"error": str(e)},
        }


# Register built-in health checks
health_checker.register_check("system_resources", check_system_resources)
health_checker.register_check("process_health", check_process_health)
health_checker.register_check("configuration", check_configuration)


# Convenience functions
async def check_health() -> SystemHealth:
    """Run all health checks."""
    return await health_checker.run_all_checks()


def is_system_healthy() -> bool:
    """Check if system is healthy based on last health check."""
    return health_checker.is_healthy()


def get_health_summary() -> dict[str, Any]:
    """Get health summary from last check."""
    last_check = health_checker.get_last_check()
    if not last_check:
        return {"status": "unknown", "message": "No health check performed"}

    return {
        "status": last_check.status.value,
        "timestamp": last_check.timestamp.isoformat(),
        "summary": last_check.summary,
        "checks": [
            {
                "name": check.name,
                "status": check.status.value,
                "message": check.message,
                "duration_ms": check.duration_ms,
            }
            for check in last_check.checks
        ],
    }
