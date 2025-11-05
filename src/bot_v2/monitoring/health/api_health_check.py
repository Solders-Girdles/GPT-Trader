"""API health check implementation."""

from __future__ import annotations

import time
from typing import Any

from .base import HealthChecker, HealthCheckResult, HealthStatus


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
