"""Concrete health check implementations."""

from __future__ import annotations

import time
from collections.abc import Callable
from numbers import Real
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from bot_v2.utilities.performance_monitoring import get_performance_health_check

from .base import HealthChecker, HealthCheckResult, HealthStatus


class DatabaseHealthCheck(HealthChecker):
    """Health check for database connectivity."""

    def __init__(self, connection_factory: Callable[[], Any], name: str = "database") -> None:
        super().__init__(name)
        self.connection_factory = connection_factory

    async def _do_check(self) -> HealthCheckResult:
        try:
            connection = self.connection_factory()
            start_time = time.time()
            await connection.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "query_time_ms": query_time,
                    "connection_type": type(connection).__name__,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class APIHealthCheck(HealthChecker):
    """Health check for external API connectivity."""

    def __init__(self, api_client: Any, endpoint: str = "/health", name: str = "api") -> None:
        super().__init__(name)
        self.api_client = api_client
        self.endpoint = endpoint

    async def _do_check(self) -> HealthCheckResult:
        try:
            start_time = time.time()
            response = await self.api_client.get(self.endpoint)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "API endpoint responsive"
            else:
                status = HealthStatus.DEGRADED
                message = f"API returned status {response.status_code}"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "endpoint": self.endpoint,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"API health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class BrokerageHealthCheck(HealthChecker):
    """Health check for brokerage connectivity."""

    def __init__(self, brokerage: Any, name: str = "brokerage") -> None:
        super().__init__(name)
        self.brokerage = brokerage

    async def _do_check(self) -> HealthCheckResult:
        try:
            if not self.brokerage.validate_connection():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Brokerage not connected",
                )

            start_time = time.time()
            balances = self.brokerage.list_balances()
            response_time = (time.time() - start_time) * 1000
            products = self.brokerage.list_products()

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Brokerage connection healthy",
                details={
                    "response_time_ms": response_time,
                    "balance_count": len(balances),
                    "product_count": len(products),
                    "account_id": self.brokerage.get_account_id(),
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Brokerage health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class MemoryHealthCheck(HealthChecker):
    """Health check for memory usage."""

    def __init__(
        self,
        warning_threshold_mb: float = 1000.0,
        critical_threshold_mb: float = 2000.0,
        name: str = "memory",
    ) -> None:
        super().__init__(name)
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb

    async def _do_check(self) -> HealthCheckResult:
        try:
            import sys

            module = sys.modules.get("bot_v2.monitoring.health_checks")
            psutil_module = getattr(module, "psutil", psutil)
            if psutil_module is None:
                raise ImportError

            try:
                process = psutil_module.Process()
            except Exception as exc:
                raise ImportError from exc
            memory_info = process.memory_info()
            rss_value = getattr(memory_info, "rss", None)
            if not isinstance(rss_value, Real):
                raise ImportError
            memory_mb = rss_value / 1024 / 1024

            if memory_mb > self.critical_threshold_mb:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_mb:.1f}MB"
            elif memory_mb > self.warning_threshold_mb:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"

            try:
                memory_percent = float(process.memory_percent())
            except Exception:
                memory_percent = 0.0

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "memory_mb": round(memory_mb, 1),
                    "warning_threshold_mb": self.warning_threshold_mb,
                    "critical_threshold_mb": self.critical_threshold_mb,
                    "memory_percent": memory_percent,
                },
            )
        except ImportError:  # pragma: no cover - optional dependency
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory monitoring",
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class PerformanceHealthCheck(HealthChecker):
    """Health check based on performance metrics."""

    def __init__(
        self,
        slow_operation_threshold_s: float = 1.0,
        very_slow_operation_threshold_s: float = 5.0,
        name: str = "performance",
    ) -> None:
        super().__init__(name)
        self.slow_operation_threshold_s = slow_operation_threshold_s
        self.very_slow_operation_threshold_s = very_slow_operation_threshold_s

    async def _do_check(self) -> HealthCheckResult:
        try:
            import sys

            module = sys.modules.get("bot_v2.monitoring.health_checks")
            perf_fn = getattr(module, "get_performance_health_check", get_performance_health_check)
            perf_health = perf_fn()
            status = HealthStatus(perf_health["status"])
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"Performance status: {perf_health['status']}",
                details={
                    "issues": perf_health["issues"],
                    "metrics": perf_health["metrics"],
                    "slow_threshold_s": self.slow_operation_threshold_s,
                    "very_slow_threshold_s": self.very_slow_operation_threshold_s,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Performance health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


__all__ = [
    "DatabaseHealthCheck",
    "APIHealthCheck",
    "BrokerageHealthCheck",
    "MemoryHealthCheck",
    "PerformanceHealthCheck",
]
