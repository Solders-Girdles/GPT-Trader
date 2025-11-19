"""Brokerage health check implementation."""

from __future__ import annotations

import time
from typing import Any

from .base import HealthChecker, HealthCheckResult, HealthStatus


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
