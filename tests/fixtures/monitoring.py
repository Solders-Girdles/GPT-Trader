"""
Monitoring-specific test helpers.

These utilities provide deterministic collectors and alert instrumentation so
monitoring tests can focus on behavioural coverage instead of wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Callable

import pytest

try:  # pragma: no cover - optional dependency shim
    from freezegun import freeze_time as _freeze_time
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency absent
    from contextlib import contextmanager

    @contextmanager
    def _freeze_time(*_args: object, **_kwargs: object):
        yield


from bot_v2.monitoring.alert_types import AlertSeverity
from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    MonitorConfig,
    PerformanceMetrics,
    ResourceUsage,
)
from bot_v2.monitoring.system.alerting import AlertManager


@dataclass
class ResourceSnapshot:
    """Convenience container for resource metrics."""

    cpu_percent: float = 10.0
    memory_percent: float = 35.0
    memory_mb: float = 512.0
    disk_percent: float = 45.0
    disk_gb: float = 120.0
    network_sent_mb: float = 1.0
    network_recv_mb: float = 1.5
    open_files: int = 4
    threads: int = 8

    def to_usage(self) -> ResourceUsage:
        return ResourceUsage(
            cpu_percent=self.cpu_percent,
            memory_percent=self.memory_percent,
            memory_mb=self.memory_mb,
            disk_percent=self.disk_percent,
            disk_gb=self.disk_gb,
            network_sent_mb=self.network_sent_mb,
            network_recv_mb=self.network_recv_mb,
            open_files=self.open_files,
            threads=self.threads,
        )


class DeterministicResourceCollector:
    """Collects a pre-configured resource snapshot."""

    def __init__(self, snapshot: ResourceSnapshot | None = None) -> None:
        self._snapshot = snapshot or ResourceSnapshot()

    def update(self, **overrides: float | int) -> None:
        """Replace snapshot values for the next collection."""
        data = self._snapshot.__dict__ | overrides
        self._snapshot = ResourceSnapshot(**data)

    def collect(self) -> ResourceUsage:
        return self._snapshot.to_usage()


@dataclass
class PerformanceSnapshot:
    """Convenience container for performance metrics."""

    requests_per_second: float = 100.0
    avg_response_time_ms: float = 50.0
    p95_response_time_ms: float = 75.0
    p99_response_time_ms: float = 90.0
    error_rate: float = 0.01
    success_rate: float = 0.99
    active_connections: int = 12
    queued_tasks: int = 1

    def to_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            requests_per_second=self.requests_per_second,
            avg_response_time_ms=self.avg_response_time_ms,
            p95_response_time_ms=self.p95_response_time_ms,
            p99_response_time_ms=self.p99_response_time_ms,
            error_rate=self.error_rate,
            success_rate=self.success_rate,
            active_connections=self.active_connections,
            queued_tasks=self.queued_tasks,
        )


class DeterministicPerformanceCollector:
    """Collects a pre-configured performance snapshot."""

    def __init__(self, snapshot: PerformanceSnapshot | None = None) -> None:
        self._snapshot = snapshot or PerformanceSnapshot()

    def update(self, **overrides: float | int) -> None:
        data = self._snapshot.__dict__ | overrides
        self._snapshot = PerformanceSnapshot(**data)

    def collect(self) -> PerformanceMetrics:
        return self._snapshot.to_metrics()

    def record_request(self, duration_ms: float, success: bool) -> None:  # noqa: D401
        """No-op hook (kept for interface compatibility)."""
        return None


class DeterministicComponentCollector:
    """Returns a stable component map, configurable per test."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentHealth] = {
            "DataProvider": ComponentHealth(
                name="DataProvider",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
                uptime_seconds=3600.0,
                error_count=0,
                warning_count=0,
                details={"latency_ms": 12.0},
            )
        }

    def set_component(
        self,
        name: str,
        *,
        status: ComponentStatus,
        error_count: int = 0,
        warning_count: int = 0,
        details: dict | None = None,
        uptime_seconds: float = 3600.0,
    ) -> None:
        self._components[name] = ComponentHealth(
            name=name,
            status=status,
            last_check=datetime.utcnow(),
            uptime_seconds=uptime_seconds,
            error_count=error_count,
            warning_count=warning_count,
            details=details or {},
        )

    def collect(self) -> dict[str, ComponentHealth]:
        return dict(self._components)


@pytest.fixture
def monitoring_collectors():
    """
    Provide deterministic collectors for MonitoringSystem tests.

    Returns a dict with ``resource``, ``performance`` and ``component`` keys so
    tests can tweak snapshots before invoking the system under test.
    """

    collectors = {
        "resource": DeterministicResourceCollector(),
        "performance": DeterministicPerformanceCollector(),
        "component": DeterministicComponentCollector(),
    }
    return collectors


class AlertRecorder:
    """Helper that captures alerts emitted by an AlertManager."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.alerts = []

    def attach(self, manager: AlertManager) -> AlertManager:
        """Patch the manager so created alerts are captured and prints muted."""

        self._monkeypatch.setattr(manager, "_print_alert", lambda alert: None)
        self._monkeypatch.setattr(manager, "_send_notification", lambda alert: None)

        original_create = manager.create_alert

        def wrapped_create_alert(*args, **kwargs):
            alert = original_create(*args, **kwargs)
            if alert is not None:
                self.alerts.append(alert)
            return alert

        self._monkeypatch.setattr(manager, "create_alert", wrapped_create_alert)
        return manager

    def ids(self) -> list[str]:
        return [alert.alert_id for alert in self.alerts]

    def by_severity(self, severity: AlertSeverity) -> list:
        return [alert for alert in self.alerts if alert.severity == severity]


@pytest.fixture
def alert_recorder(monkeypatch: pytest.MonkeyPatch):
    """Fixture exposing an AlertRecorder helper."""
    return AlertRecorder(monkeypatch)


@pytest.fixture
def alert_manager(alert_recorder) -> AlertManager:
    """
    Provide an AlertManager with notifications disabled and recording enabled.

    Tests can still customise thresholds via config as needed.
    """
    config = MonitorConfig(enable_notifications=False, check_interval_seconds=1)
    manager = AlertManager(config)
    alert_recorder.attach(manager)
    return manager


@pytest.fixture
def frozen_time():
    """
    Freeze ``datetime.now`` during a test.

    Returns the freezegun freezer so callers can ``move_to`` as scenarios evolve.
    """
    with _freeze_time("2025-01-01 12:00:00") as freezer:
        yield freezer


def advance_time(seconds: int, freezer) -> None:
    """Advance a freezegun freezer by ``seconds`` to emulate elapsed time."""
    freezer.tick(delta=timedelta(seconds=seconds))


__all__ = [
    "alert_manager",
    "alert_recorder",
    "advance_time",
    "frozen_time",
    "monitoring_collectors",
    "DeterministicComponentCollector",
    "DeterministicPerformanceCollector",
    "DeterministicResourceCollector",
    "PerformanceSnapshot",
    "ResourceSnapshot",
]
