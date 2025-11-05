"""Memory health check implementation."""

from __future__ import annotations

from numbers import Real
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from .base import HealthChecker, HealthCheckResult, HealthStatus


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
            memory_mb = float(rss_value) / (1024 * 1024)

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
