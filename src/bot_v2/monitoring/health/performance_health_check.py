"""Performance health check implementation."""

from __future__ import annotations

from bot_v2.utilities.performance_monitoring import get_performance_health_check

from .base import HealthChecker, HealthCheckResult, HealthStatus


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
