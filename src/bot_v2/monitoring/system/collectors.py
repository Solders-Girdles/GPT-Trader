"""
Local metric collectors for monitoring.

Complete isolation - no external dependencies.
"""

import random
from datetime import datetime
from typing import Any

import psutil

from ..interfaces import ComponentHealth, ComponentStatus, PerformanceMetrics, ResourceUsage


class ResourceCollector:
    """Collects system resource metrics."""

    def __init__(self) -> None:
        self.last_network: Any | None = None
        self.last_collect_time: datetime | None = None

    def collect(self) -> ResourceUsage:
        """Collect current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
        except Exception:
            cpu_percent = 0.0

        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
        except Exception:
            memory_percent = 0.0
            memory_mb = 0.0

        try:
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_gb = disk.used / (1024 * 1024 * 1024)
        except Exception:
            disk_percent = 0.0
            disk_gb = 0.0

        try:
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
        except Exception:
            network_sent_mb = 0.0
            network_recv_mb = 0.0

        try:
            process = psutil.Process()
            open_files = len(process.open_files()) if hasattr(process, "open_files") else 0
            threads = process.num_threads()
        except Exception:
            open_files = 0
            threads = 0

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_percent=disk_percent,
            disk_gb=disk_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            open_files=open_files,
            threads=threads,
        )


class PerformanceCollector:
    """Collects performance metrics."""

    def __init__(self) -> None:
        self.request_times: list[float] = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = datetime.now()

    def collect(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Simulated metrics (would be real in production)
        # Add some randomness for realistic monitoring
        base_response_time = 50
        variance = random.uniform(0.8, 1.2)

        avg_response = base_response_time * variance
        p95_response = avg_response * 1.5
        p99_response = avg_response * 2.0

        # Simulate occasional errors
        error_rate = 0.01 if random.random() > 0.1 else 0.05
        success_rate = 1 - error_rate

        # Requests per second (simulated)
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
        """
        Record a request for metrics.

        Args:
            duration_ms: Request duration in milliseconds
            success: Whether request was successful
        """
        self.request_times.append(duration_ms)
        if len(self.request_times) > 1000:
            self.request_times.pop(0)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1


class ComponentCollector:
    """Collects component health status."""

    def __init__(self) -> None:
        self.component_start_times: dict[str, datetime] = {}
        self.component_errors: dict[str, int] = {}
        self.component_warnings: dict[str, int] = {}

    def collect(self) -> dict[str, ComponentHealth]:
        """Collect health status of all components."""
        components: dict[str, ComponentHealth] = {}

        # Check each major component
        components["DataProvider"] = self._check_data_provider()
        components["Strategies"] = self._check_strategies()
        components["RiskManager"] = self._check_risk_manager()
        components["Executor"] = self._check_executor()
        components["Database"] = self._check_database()
        components["API"] = self._check_api()

        return components

    def _check_data_provider(self) -> ComponentHealth:
        """Check data provider health."""
        # Simulated check
        is_healthy = random.random() > 0.05  # 95% healthy

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
        """Check strategies health."""
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
        """Check risk manager health."""
        return ComponentHealth(
            name="RiskManager",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("RiskManager"),
            error_count=self.component_errors.get("RiskManager", 0),
            warning_count=self.component_warnings.get("RiskManager", 0),
            details={
                "rules_active": 12,
                "violations_today": random.randint(0, 5),
                "positions_monitored": random.randint(0, 10),
            },
        )

    def _check_executor(self) -> ComponentHealth:
        """Check executor health."""
        is_healthy = random.random() > 0.02  # 98% healthy

        return ComponentHealth(
            name="Executor",
            status=ComponentStatus.HEALTHY if is_healthy else ComponentStatus.DEGRADED,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("Executor"),
            error_count=self.component_errors.get("Executor", 0),
            warning_count=self.component_warnings.get("Executor", 0),
            details={
                "orders_pending": random.randint(0, 5),
                "orders_today": random.randint(20, 100),
                "avg_execution_ms": random.uniform(50, 200),
            },
        )

    def _check_database(self) -> ComponentHealth:
        """Check database health."""
        return ComponentHealth(
            name="Database",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("Database"),
            error_count=self.component_errors.get("Database", 0),
            warning_count=self.component_warnings.get("Database", 0),
            details={
                "connections_active": random.randint(5, 20),
                "queries_per_second": random.uniform(10, 100),
                "storage_gb": random.uniform(1, 10),
            },
        )

    def _check_api(self) -> ComponentHealth:
        """Check API health."""
        is_healthy = random.random() > 0.01  # 99% healthy

        return ComponentHealth(
            name="API",
            status=ComponentStatus.HEALTHY if is_healthy else ComponentStatus.UNHEALTHY,
            last_check=datetime.now(),
            uptime_seconds=self._get_uptime("API"),
            error_count=self.component_errors.get("API", 0),
            warning_count=self.component_warnings.get("API", 0),
            details={
                "endpoints_active": 15,
                "requests_per_minute": random.randint(100, 1000),
                "avg_latency_ms": random.uniform(20, 100),
            },
        )

    def _get_uptime(self, component: str) -> float:
        """Get component uptime in seconds."""
        if component not in self.component_start_times:
            self.component_start_times[component] = datetime.now()

        uptime = datetime.now() - self.component_start_times[component]
        return uptime.total_seconds()
