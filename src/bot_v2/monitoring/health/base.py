"""Core health-check primitives."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger("health", component="monitoring")


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.status.value} - {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class HealthChecker:
    """Base class for health check implementations."""

    def __init__(self, name: str, timeout: float = 10.0) -> None:
        self.name = name
        self.timeout = timeout

    async def check_health(self) -> HealthCheckResult:
        start_time = time.time()
        try:
            result = await asyncio.wait_for(self._do_check(), timeout=self.timeout)
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Health check failed", name=self.name, error=str(exc))
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {exc}",
                details={"error_type": type(exc).__name__},
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _do_check(self) -> HealthCheckResult:
        raise NotImplementedError


__all__ = ["HealthStatus", "HealthCheckResult", "HealthChecker", "logger"]
