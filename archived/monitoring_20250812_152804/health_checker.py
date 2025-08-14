"""
Comprehensive health monitoring system for GPT-Trader.

Provides health checks for:
- System resources
- Service dependencies
- Database connectivity
- API endpoints
- Background jobs
- Data freshness
"""

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psutil


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    checks: list[HealthCheck]
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": [check.to_dict() for check in self.checks],
        }


class HealthChecker:
    """System health monitoring."""

    def __init__(self, check_interval: int = 30):
        """
        Initialize health checker.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.checks = {}
        self.last_check_time = {}
        self.check_history = {}
        self.start_time = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Background checker
        self._stop_event = threading.Event()
        self._check_thread = None

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)
        self.register_check("process", self._check_process)

    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check."""
        with self._lock:
            self.checks[name] = check_func
            self.check_history[name] = []

    def start(self):
        """Start background health checking."""
        if self._check_thread is None:
            self._check_thread = threading.Thread(target=self._background_check, daemon=True)
            self._check_thread.start()

    def stop(self):
        """Stop background health checking."""
        self._stop_event.set()
        if self._check_thread:
            self._check_thread.join(timeout=5)

    def _background_check(self):
        """Background thread for health checks."""
        while not self._stop_event.is_set():
            self.run_all_checks()
            time.sleep(self.check_interval)

    def run_check(self, name: str) -> HealthCheck | None:
        """Run a specific health check."""
        with self._lock:
            if name not in self.checks:
                return None

            start_time = time.perf_counter()

            try:
                result = self.checks[name]()
                result.duration_ms = (time.perf_counter() - start_time) * 1000

                # Update history
                self.check_history[name].append(result)
                if len(self.check_history[name]) > 100:
                    self.check_history[name] = self.check_history[name][-50:]

                self.last_check_time[name] = datetime.now()

                return result
            except Exception as e:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )

    def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        checks = []

        with self._lock:
            for name in self.checks:
                result = self.run_check(name)
                if result:
                    checks.append(result)

        # Determine overall status
        statuses = [check.status for check in checks]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status, checks=checks, uptime_seconds=time.time() - self.start_time
        )

    def get_status(self) -> SystemHealth:
        """Get current health status."""
        return self.run_all_checks()

    # Default health check implementations

    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "cpu_count": psutil.cpu_count(),
        }

        if cpu_percent > 90:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"High CPU usage: {cpu_percent}%",
                details=details,
            )
        elif cpu_percent > 75 or memory_percent > 85:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.DEGRADED,
                message="Elevated resource usage",
                details=details,
            )
        else:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.HEALTHY,
                message="Resources within normal range",
                details=details,
            )

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        disk_usage = psutil.disk_usage("/")
        percent_used = disk_usage.percent
        free_gb = disk_usage.free / (1024**3)

        details = {
            "percent_used": percent_used,
            "free_gb": round(free_gb, 2),
            "total_gb": round(disk_usage.total / (1024**3), 2),
        }

        if percent_used > 95 or free_gb < 1:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Critical disk space: {percent_used}% used",
                details=details,
            )
        elif percent_used > 85 or free_gb < 5:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Low disk space: {free_gb:.1f}GB free",
                details=details,
            )
        else:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                message=f"Adequate disk space: {free_gb:.1f}GB free",
                details=details,
            )

    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        memory = psutil.virtual_memory()

        details = {
            "percent_used": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2),
            "swap_percent": psutil.swap_memory().percent,
        }

        if memory.percent > 95:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Critical memory usage: {memory.percent}%",
                details=details,
            )
        elif memory.percent > 85:
            return HealthCheck(
                name="memory",
                status=HealthStatus.DEGRADED,
                message=f"High memory usage: {memory.percent}%",
                details=details,
            )
        else:
            return HealthCheck(
                name="memory",
                status=HealthStatus.HEALTHY,
                message=f"Memory usage normal: {memory.percent}%",
                details=details,
            )

    def _check_process(self) -> HealthCheck:
        """Check current process health."""
        process = psutil.Process(os.getpid())

        details = {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
        }

        if details["cpu_percent"] > 90 or details["memory_mb"] > 2048:
            return HealthCheck(
                name="process",
                status=HealthStatus.UNHEALTHY,
                message="Process resource usage critical",
                details=details,
            )
        elif details["cpu_percent"] > 75 or details["memory_mb"] > 1024:
            return HealthCheck(
                name="process",
                status=HealthStatus.DEGRADED,
                message="Process resource usage elevated",
                details=details,
            )
        else:
            return HealthCheck(
                name="process",
                status=HealthStatus.HEALTHY,
                message="Process healthy",
                details=details,
            )


class ServiceHealthChecker(HealthChecker):
    """Extended health checker for services."""

    def __init__(self, check_interval: int = 30):
        """Initialize service health checker."""
        super().__init__(check_interval)

        # Register service checks
        self.register_check("database", self._check_database)
        self.register_check("api", self._check_api)
        self.register_check("data_feed", self._check_data_feed)

    def _check_database(self) -> HealthCheck:
        """Check database connectivity."""
        try:
            # Simulated database check
            # In production, would execute a simple query
            import random

            latency_ms = random.uniform(1, 50)

            details = {"latency_ms": round(latency_ms, 2), "connections": 5, "pool_size": 10}

            if latency_ms > 100:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database response time critical",
                    details=details,
                )
            elif latency_ms > 50:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message="Database response time elevated",
                    details=details,
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database responding normally",
                    details=details,
                )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
            )

    def _check_api(self) -> HealthCheck:
        """Check API endpoints."""
        try:
            # Simulated API check
            import random

            response_time = random.uniform(10, 200)
            error_rate = random.uniform(0, 5)

            details = {
                "response_time_ms": round(response_time, 2),
                "error_rate_percent": round(error_rate, 2),
                "requests_per_second": random.randint(10, 100),
            }

            if response_time > 1000 or error_rate > 10:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.UNHEALTHY,
                    message="API performance critical",
                    details=details,
                )
            elif response_time > 500 or error_rate > 5:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.DEGRADED,
                    message="API performance degraded",
                    details=details,
                )
            else:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.HEALTHY,
                    message="API operating normally",
                    details=details,
                )
        except Exception as e:
            return HealthCheck(
                name="api", status=HealthStatus.UNHEALTHY, message=f"API check failed: {str(e)}"
            )

    def _check_data_feed(self) -> HealthCheck:
        """Check market data feed."""
        try:
            # Simulated data feed check
            import random

            last_update = datetime.now() - timedelta(seconds=random.randint(1, 120))
            lag_seconds = (datetime.now() - last_update).total_seconds()

            details = {
                "last_update": last_update.isoformat(),
                "lag_seconds": lag_seconds,
                "symbols_tracked": 50,
                "updates_per_second": random.randint(10, 100),
            }

            if lag_seconds > 60:
                return HealthCheck(
                    name="data_feed",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Data feed stale: {lag_seconds}s behind",
                    details=details,
                )
            elif lag_seconds > 30:
                return HealthCheck(
                    name="data_feed",
                    status=HealthStatus.DEGRADED,
                    message=f"Data feed lagging: {lag_seconds}s behind",
                    details=details,
                )
            else:
                return HealthCheck(
                    name="data_feed",
                    status=HealthStatus.HEALTHY,
                    message="Data feed up to date",
                    details=details,
                )
        except Exception as e:
            return HealthCheck(
                name="data_feed",
                status=HealthStatus.UNHEALTHY,
                message=f"Data feed check failed: {str(e)}",
            )


def create_health_endpoint(checker: HealthChecker):
    """Create health check endpoint response."""
    health = checker.get_status()

    # Format for HTTP response
    response = {
        "status": health.status.value,
        "timestamp": health.timestamp.isoformat(),
        "version": health.version,
        "uptime": health.uptime_seconds,
        "checks": {},
    }

    for check in health.checks:
        response["checks"][check.name] = {
            "status": check.status.value,
            "message": check.message,
            "duration_ms": check.duration_ms,
        }

        if check.details:
            response["checks"][check.name]["details"] = check.details

    # Determine HTTP status code
    if health.status == HealthStatus.HEALTHY:
        status_code = 200
    elif health.status == HealthStatus.DEGRADED:
        status_code = 200  # Still operational
    else:
        status_code = 503  # Service unavailable

    return response, status_code


def demo_health_monitoring():
    """Demonstrate health monitoring."""
    print("Health Monitoring Demo")
    print("=" * 50)

    # Create health checker
    checker = ServiceHealthChecker(check_interval=5)
    checker.start()

    # Run initial health check
    print("\nInitial Health Check:")
    health = checker.get_status()

    print(f"Overall Status: {health.status.value.upper()}")
    print(f"Uptime: {health.uptime_seconds:.1f} seconds")

    print("\nIndividual Checks:")
    for check in health.checks:
        status_icon = (
            "✓"
            if check.status == HealthStatus.HEALTHY
            else "⚠"
            if check.status == HealthStatus.DEGRADED
            else "✗"
        )
        print(f"  {status_icon} {check.name}: {check.status.value}")
        if check.message:
            print(f"     {check.message}")

    # Simulate health endpoint
    print("\nHealth Endpoint Response:")
    response, status_code = create_health_endpoint(checker)
    print(f"HTTP {status_code}")
    print(json.dumps(response, indent=2)[:500] + "...")

    # Stop checker
    checker.stop()

    print("\n✓ Health monitoring demo complete")

    return checker


if __name__ == "__main__":
    checker = demo_health_monitoring()
