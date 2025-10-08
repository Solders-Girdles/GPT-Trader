"""
Local types for system monitoring.

Complete isolation - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from bot_v2.monitoring.alert_types import Alert as AlertType
from bot_v2.monitoring.alert_types import AlertSeverity

# Backwards-compatible alias – code should transition to ``AlertSeverity``
AlertLevel = AlertSeverity
Alert = AlertType


class ComponentStatus(Enum):
    """Component health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ResourceUsage:
    """System resource usage metrics."""

    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    disk_gb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int

    def is_high_usage(self) -> bool:
        """Check if resource usage is high."""
        return self.cpu_percent > 80 or self.memory_percent > 80 or self.disk_percent > 90


@dataclass
class PerformanceMetrics:
    """System performance metrics."""

    requests_per_second: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    success_rate: float
    active_connections: int
    queued_tasks: int

    def is_degraded(self) -> bool:
        """Check if performance is degraded."""
        return (
            self.avg_response_time_ms > 1000 or self.error_rate > 0.05 or self.success_rate < 0.95
        )


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    name: str
    status: ComponentStatus
    last_check: datetime
    uptime_seconds: float
    error_count: int
    warning_count: int
    details: dict[str, Any]

    def get_uptime_hours(self) -> float:
        """Get uptime in hours."""
        return self.uptime_seconds / 3600


@dataclass
class SystemHealth:
    """Overall system health status."""

    timestamp: datetime
    overall_status: ComponentStatus
    components: dict[str, ComponentHealth]
    resource_usage: ResourceUsage
    performance: PerformanceMetrics
    active_alerts: int

    def summary(self) -> str:
        """Generate health summary."""
        healthy_count = sum(
            1 for c in self.components.values() if c.status == ComponentStatus.HEALTHY
        )
        total_count = len(self.components)

        return f"""
System Health Report
====================
Timestamp: {self.timestamp}
Overall Status: {self.overall_status.value.upper()}
Components: {healthy_count}/{total_count} healthy
Active Alerts: {self.active_alerts}

Resource Usage:
- CPU: {self.resource_usage.cpu_percent:.1f}%
- Memory: {self.resource_usage.memory_percent:.1f}%
- Disk: {self.resource_usage.disk_percent:.1f}%

Performance:
- Avg Response: {self.performance.avg_response_time_ms:.1f}ms
- Error Rate: {self.performance.error_rate:.2%}
- Success Rate: {self.performance.success_rate:.2%}
        """.strip()


@dataclass
class MonitorConfig:
    """Monitoring configuration."""

    check_interval_seconds: int = 60
    alert_threshold_cpu: float = 80.0
    alert_threshold_memory: float = 80.0
    alert_threshold_disk: float = 90.0
    alert_threshold_error_rate: float = 0.05
    alert_threshold_response_time_ms: float = 1000.0
    retention_days: int = 7
    enable_notifications: bool = True


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    endpoint: str | None
    check_type: Literal["http", "tcp", "process", "custom"]
    timeout_seconds: int = 30
    retry_count: int = 3
    expected_status: int | None = 200


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""

    timestamp: datetime
    metric_name: str
    value: float
    tags: dict[str, str]


@dataclass
class TradeMetrics:
    """Trading-specific metrics."""

    total_trades_today: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_pnl: float
    win_rate: float
    avg_execution_time_ms: float
    slippage_bps: float  # Basis points
