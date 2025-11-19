"""Database health check implementation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

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
