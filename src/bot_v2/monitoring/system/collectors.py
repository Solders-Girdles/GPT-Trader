"""Interfaces and reusable collectors for the monitoring system."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    PerformanceMetrics,
    ResourceUsage,
)
from bot_v2.utilities.logging_patterns import get_logger

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - dependency optional
    psutil = None  # type: ignore[assignment]

logger = get_logger(__name__, component="monitoring_collectors")


class ResourceCollector:
    """Collects resource metrics from the local machine using ``psutil``."""

    _warning_logged = False

    def __init__(self) -> None:
        self._psutil = psutil
        if self._psutil is None and not ResourceCollector._warning_logged:
            logger.warning(
                "psutil not installed; system metrics will return zero values. "
                "Install with `pip install gpt-trader[monitoring]` to enable resource collection.",
                operation="resource_collector_init",
                stage="dependency_missing",
            )
            ResourceCollector._warning_logged = True

    def collect(self) -> ResourceUsage:
        if self._psutil is None:
            return ResourceUsage(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_percent=0.0,
                disk_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                open_files=0,
                threads=0,
            )

        try:
            cpu_percent = self._psutil.cpu_percent(interval=0.0)
        except Exception:
            cpu_percent = 0.0

        try:
            memory = self._psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
        except Exception:
            memory_percent = 0.0
            memory_mb = 0.0

        try:
            disk = self._psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_gb = disk.used / (1024 * 1024 * 1024)
        except Exception:
            disk_percent = 0.0
            disk_gb = 0.0

        try:
            network = self._psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
        except Exception:
            network_sent_mb = 0.0
            network_recv_mb = 0.0

        try:
            process = self._psutil.Process()
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


class PerformanceCollector(Protocol):
    """Behaviour expected from performance collectors."""

    def collect(self) -> PerformanceMetrics:
        """Return a snapshot of performance metrics."""

    def record_request(self, duration_ms: float, success: bool) -> None:
        """Track an individual request to enrich metrics."""


class ComponentCollector(Protocol):
    """Behaviour expected from component health collectors."""

    def collect(self) -> dict[str, ComponentHealth]:
        """Return per-component health information."""


class NullComponentCollector:
    """Fallback collector used when no component collector is supplied."""

    def collect(self) -> dict[str, ComponentHealth]:
        now = datetime.now()
        return {
            "system": ComponentHealth(
                name="system",
                status=ComponentStatus.UNKNOWN,
                last_check=now,
                uptime_seconds=0.0,
                error_count=0,
                warning_count=0,
                details={"note": "No component collector configured"},
            )
        }


class NullPerformanceCollector:
    """Fallback performance collector that reports empty metrics."""

    def collect(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            requests_per_second=0.0,
            avg_response_time_ms=0.0,
            p95_response_time_ms=0.0,
            p99_response_time_ms=0.0,
            error_rate=0.0,
            success_rate=1.0,
            active_connections=0,
            queued_tasks=0,
        )

    def record_request(self, duration_ms: float, success: bool) -> None:  # noqa: D401
        """No-op placeholder."""
        return None


__all__ = [
    "ResourceCollector",
    "PerformanceCollector",
    "ComponentCollector",
    "NullComponentCollector",
    "NullPerformanceCollector",
]
