"""Demo collectors that generate synthetic monitoring data.

These implementations keep the sandbox/dry-run experience intact while real
collectors are wired up. Production deployments should supply collectors that
query live systems instead of relying on this module.
"""

from __future__ import annotations

import random
from datetime import datetime

from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    PerformanceMetrics,
)
from bot_v2.monitoring.system.collectors import ComponentCollector, PerformanceCollector


class DemoPerformanceCollector(PerformanceCollector):
    """Generates plausible performance metrics without external dependencies."""

    def __init__(self) -> None:
        self.request_times: list[float] = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = datetime.now()

    def collect(self) -> PerformanceMetrics:
        base_response_time = 50
        variance = random.uniform(0.8, 1.2)

        avg_response = base_response_time * variance
        p95_response = avg_response * 1.5
        p99_response = avg_response * 2.0

        error_rate = 0.01 if random.random() > 0.1 else 0.05
        success_rate = 1 - error_rate

        rps = 100 * variance

        return PerformanceMetrics(
            requests_per_second=rps,
            avg_response_time_ms=avg_response,
            p95_response_time_ms=p95_response,
            p99_response_time_ms=p99_response,
            error_rate=error_rate,
            success_rate=success_rate,
            active_connections=random.randint(10, 50),
            queued_tasks=random.randint(0, 10),
        )

    def record_request(self, duration_ms: float, success: bool) -> None:
        self.request_times.append(duration_ms)
        if len(self.request_times) > 1000:
            self.request_times.pop(0)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1


class DemoComponentCollector(ComponentCollector):
    """Produces canned component health information for demos."""

    def __init__(self) -> None:
        self.component_start_times: dict[str, datetime] = {}
        self.component_errors: dict[str, int] = {}
        self.component_warnings: dict[str, int] = {}

    def collect(self) -> dict[str, ComponentHealth]:
        components: dict[str, ComponentHealth] = {}
        components["DataProvider"] = self._check_data_provider()
        components["Strategies"] = self._check_strategies()
        components["RiskManager"] = self._check_risk_manager()
        components["Executor"] = self._check_executor()
        components["Database"] = self._check_database()
        components["API"] = self._check_api()
        return components

    def _check_data_provider(self) -> ComponentHealth:
        is_healthy = random.random() > 0.05
        return ComponentHealth(
            name="DataProvider",
            status=ComponentStatus.HEALTHY if is_healthy else ComponentStatus.DEGRADED,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("DataProvider"),
            error_count=self.component_errors.get("DataProvider", 0),
            warning_count=self.component_warnings.get("DataProvider", 0),
            details={
                "data_sources": 3,
                "last_update": datetime.now().isoformat(),
                "latency_ms": random.uniform(10, 50),
            },
        )

    def _check_strategies(self) -> ComponentHealth:
        return ComponentHealth(
            name="Strategies",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("Strategies"),
            error_count=self.component_errors.get("Strategies", 0),
            warning_count=self.component_warnings.get("Strategies", 0),
            details={
                "active_strategies": 5,
                "signals_generated": random.randint(10, 100),
                "avg_calculation_ms": random.uniform(5, 20),
            },
        )

    def _check_risk_manager(self) -> ComponentHealth:
        return ComponentHealth(
            name="RiskManager",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("RiskManager"),
            error_count=self.component_errors.get("RiskManager", 0),
            warning_count=self.component_warnings.get("RiskManager", 0),
            details={
                "checks_passed": random.randint(95, 100),
                "risk_limit": "Normal",
                "violations": 0,
            },
        )

    def _check_executor(self) -> ComponentHealth:
        return ComponentHealth(
            name="Executor",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("Executor"),
            error_count=self.component_errors.get("Executor", 0),
            warning_count=self.component_warnings.get("Executor", 0),
            details={
                "orders_processed": random.randint(100, 1000),
                "fills": random.randint(50, 200),
                "latency_ms": random.uniform(1, 5),
            },
        )

    def _check_database(self) -> ComponentHealth:
        return ComponentHealth(
            name="Database",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("Database"),
            error_count=self.component_errors.get("Database", 0),
            warning_count=self.component_warnings.get("Database", 0),
            details={
                "connections": random.randint(5, 20),
                "queries_per_sec": random.uniform(50, 150),
                "replication_lag": random.uniform(0, 1),
            },
        )

    def _check_api(self) -> ComponentHealth:
        return ComponentHealth(
            name="API",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("API"),
            error_count=self.component_errors.get("API", 0),
            warning_count=self.component_warnings.get("API", 0),
            details={
                "requests_per_sec": random.uniform(10, 100),
                "error_rate": random.uniform(0, 0.05),
                "avg_latency_ms": random.uniform(20, 80),
            },
        )

    def _get_uptime(self, component: str) -> float:
        if component not in self.component_start_times:
            self.component_start_times[component] = datetime.now()
        start_time = self.component_start_times[component]
        return (datetime.now() - start_time).total_seconds()


__all__ = ["DemoPerformanceCollector", "DemoComponentCollector"]
