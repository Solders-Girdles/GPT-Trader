"""HTTP and orchestration helpers for health checks."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .base import HealthStatus
from .checks import (
    APIHealthCheck,
    BrokerageHealthCheck,
    DatabaseHealthCheck,
    MemoryHealthCheck,
    PerformanceHealthCheck,
)
from .registry import HealthCheckRegistry, get_health_registry


class HealthCheckEndpoint:
    """HTTP endpoint wrapper around a health registry."""

    def __init__(
        self, registry: HealthCheckRegistry | None = None, include_details: bool = True
    ) -> None:
        self.registry = registry or get_health_registry()
        self.include_details = include_details

    async def get_health_status(self, check_name: str | None = None) -> dict[str, Any]:
        if check_name:
            result = await self.registry.run_check(check_name)
            if not result:
                return {
                    "status": "unknown",
                    "message": f"Health check '{check_name}' not found",
                    "timestamp": time.time(),
                }
            checks = [result.to_dict()]
            overall_status = result.status.value
        else:
            results = await self.registry.run_all_checks()
            if not results:
                return {
                    "status": "unknown",
                    "message": "No health checks configured",
                    "timestamp": time.time(),
                }
            checks = [r.to_dict() for r in results]
            if any(r.status == HealthStatus.UNHEALTHY for r in results):
                overall_status = "unhealthy"
            elif any(r.status == HealthStatus.DEGRADED for r in results):
                overall_status = "degraded"
            else:
                overall_status = "healthy"

        response = {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": checks,
        }

        if self.include_details:
            response["summary"] = {
                "total_checks": len(checks),
                "healthy": sum(1 for c in checks if c["status"] == "healthy"),
                "degraded": sum(1 for c in checks if c["status"] == "degraded"),
                "unhealthy": sum(1 for c in checks if c["status"] == "unhealthy"),
                "unknown": sum(1 for c in checks if c["status"] == "unknown"),
            }
        return response

    async def get_liveness(self) -> dict[str, Any]:
        return {"status": "alive", "timestamp": time.time()}

    async def get_readiness(self) -> dict[str, Any]:
        critical = ["database", "brokerage"]
        results = [r for name in critical if (r := await self.registry.run_check(name))]
        status = (
            "ready"
            if results and all(r.status == HealthStatus.HEALTHY for r in results)
            else "not_ready"
        )
        return {
            "status": status,
            "timestamp": time.time(),
            "checks": [r.to_dict() for r in results],
        }


def setup_basic_health_checks(
    database_connection: Any | None = None,
    brokerage: Any | None = None,
    api_client: Any | None = None,
) -> None:
    """Populate the global registry with the common health checks."""
    registry = get_health_registry()

    registry.register(MemoryHealthCheck())
    registry.register(PerformanceHealthCheck())

    if database_connection:
        registry.register(DatabaseHealthCheck(connection_factory=lambda: database_connection))
    if brokerage:
        registry.register(BrokerageHealthCheck(brokerage))
    if api_client:
        registry.register(APIHealthCheck(api_client))


def get_health_summary() -> dict[str, Any]:
    """Run health checks synchronously and return a summary."""
    registry = get_health_registry()
    endpoint = HealthCheckEndpoint(registry, include_details=False)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():  # pragma: no cover - rarely hit in tests
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, endpoint.get_health_status())
                return future.result(timeout=10.0)
        return asyncio.run(endpoint.get_health_status())
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "unknown",
            "message": f"Health check failed: {exc}",
            "timestamp": time.time(),
        }


__all__ = ["HealthCheckEndpoint", "setup_basic_health_checks", "get_health_summary"]
