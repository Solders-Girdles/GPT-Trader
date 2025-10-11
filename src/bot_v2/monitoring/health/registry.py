"""Registry for health check management."""

from __future__ import annotations

import asyncio

from .base import HealthChecker, HealthCheckResult, HealthStatus, logger


class HealthCheckRegistry:
    """Registry for managing health checks."""

    def __init__(self) -> None:
        self._checkers: dict[str, HealthChecker] = {}

    def register(self, checker: HealthChecker) -> None:
        self._checkers[checker.name] = checker
        logger.info("Registered health check", name=checker.name)

    def unregister(self, name: str) -> None:
        if name in self._checkers:
            del self._checkers[name]
            logger.info("Unregistered health check", name=name)

    def get_checker(self, name: str) -> HealthChecker | None:
        return self._checkers.get(name)

    def list_checkers(self) -> list[str]:
        return list(self._checkers.keys())

    async def run_all_checks(self) -> list[HealthCheckResult]:
        if not self._checkers:
            logger.warning("No health checks registered")
            return []

        tasks = [checker.check_health() for checker in self._checkers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results: list[HealthCheckResult] = []
        for checker_name, result in zip(self._checkers.keys(), results):
            if isinstance(result, Exception):
                health_results.append(
                    HealthCheckResult(
                        name=checker_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed with exception: {result}",
                        details={"error_type": type(result).__name__},
                    )
                )
            else:
                if (
                    result.status == HealthStatus.UNHEALTHY
                    and "error_type" in result.details
                    and "exception" not in str(result.details["error_type"]).lower()
                ):
                    result.details["error_type"] = f"{result.details['error_type']}Exception"
                health_results.append(result)

        healthy = sum(1 for r in health_results if r.status == HealthStatus.HEALTHY)
        degraded = sum(1 for r in health_results if r.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for r in health_results if r.status == HealthStatus.UNHEALTHY)

        logger.info(
            "Health checks completed",
            healthy=healthy,
            degraded=degraded,
            unhealthy=unhealthy,
        )

        return health_results

    async def run_check(self, name: str) -> HealthCheckResult | None:
        checker = self.get_checker(name)
        if not checker:
            logger.warning("Health check not found", name=name)
            return None
        return await checker.check_health()


_GLOBAL_REGISTRY = HealthCheckRegistry()


def get_health_registry() -> HealthCheckRegistry:
    """Return the singleton health registry."""

    return _GLOBAL_REGISTRY


__all__ = ["HealthCheckRegistry", "get_health_registry"]
