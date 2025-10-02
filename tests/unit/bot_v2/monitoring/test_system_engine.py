from __future__ import annotations

import types
from datetime import datetime
from typing import Any

import pytest

from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    MonitorConfig,
    PerformanceMetrics,
    ResourceUsage,
    SystemHealth,
)
from bot_v2.monitoring.system.engine import MonitoringSystem


def make_resources(cpu: float = 10.0, memory: float = 10.0, disk: float = 10.0) -> ResourceUsage:
    return ResourceUsage(
        cpu_percent=cpu,
        memory_percent=memory,
        memory_mb=512.0,
        disk_percent=disk,
        disk_gb=10.0,
        network_sent_mb=1.0,
        network_recv_mb=1.0,
        open_files=10,
        threads=5,
    )


def make_performance(
    *,
    avg_response_ms: float = 50.0,
    error_rate: float = 0.01,
    success_rate: float = 0.99,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        requests_per_second=100.0,
        avg_response_time_ms=avg_response_ms,
        p95_response_time_ms=avg_response_ms * 1.5,
        p99_response_time_ms=avg_response_ms * 2,
        error_rate=error_rate,
        success_rate=success_rate,
        active_connections=10,
        queued_tasks=5,
    )


def make_component(status: ComponentStatus) -> ComponentHealth:
    return ComponentHealth(
        name="component",
        status=status,
        last_check=datetime.now(),
        uptime_seconds=3600,
        error_count=0,
        warning_count=0,
        details={},
    )


@pytest.fixture
def monitoring_system() -> MonitoringSystem:
    config = MonitorConfig(
        alert_threshold_cpu=70.0,
        alert_threshold_memory=70.0,
        alert_threshold_disk=80.0,
        alert_threshold_error_rate=0.02,
        alert_threshold_response_time_ms=500.0,
    )
    return MonitoringSystem(config=config)


def test_determine_overall_status_handles_unhealthy(monitoring_system: MonitoringSystem) -> None:
    components = {
        "db": make_component(ComponentStatus.UNHEALTHY),
        "api": make_component(ComponentStatus.HEALTHY),
    }
    status = monitoring_system._determine_overall_status(
        components,
        make_resources(),
        make_performance(),
    )
    assert status is ComponentStatus.UNHEALTHY


def test_determine_overall_status_high_resource_usage(monitoring_system: MonitoringSystem) -> None:
    components = {"api": make_component(ComponentStatus.HEALTHY)}
    status = monitoring_system._determine_overall_status(
        components,
        make_resources(cpu=95.0),
        make_performance(),
    )
    assert status is ComponentStatus.DEGRADED


def test_determine_overall_status_performance_degraded(monitoring_system: MonitoringSystem) -> None:
    components = {"api": make_component(ComponentStatus.HEALTHY)}
    status = monitoring_system._determine_overall_status(
        components,
        make_resources(),
        make_performance(avg_response_ms=1500.0, error_rate=0.1, success_rate=0.8),
    )
    assert status is ComponentStatus.DEGRADED


def test_determine_overall_status_component_degraded(monitoring_system: MonitoringSystem) -> None:
    components = {
        "api": make_component(ComponentStatus.DEGRADED),
        "db": make_component(ComponentStatus.HEALTHY),
    }
    status = monitoring_system._determine_overall_status(
        components,
        make_resources(),
        make_performance(),
    )
    assert status is ComponentStatus.DEGRADED


def test_determine_overall_status_healthy(monitoring_system: MonitoringSystem) -> None:
    components = {"api": make_component(ComponentStatus.HEALTHY)}
    status = monitoring_system._determine_overall_status(
        components,
        make_resources(),
        make_performance(),
    )
    assert status is ComponentStatus.HEALTHY


def build_system_health(config: MonitorConfig, *, cpu: float, memory: float, disk: float, error_rate: float, response_ms: float) -> SystemHealth:
    resources = make_resources(cpu=cpu, memory=memory, disk=disk)
    performance = make_performance(avg_response_ms=response_ms, error_rate=error_rate)
    components = {"api": make_component(ComponentStatus.HEALTHY)}
    return SystemHealth(
        timestamp=datetime.now(),
        overall_status=ComponentStatus.HEALTHY,
        components=components,
        resource_usage=resources,
        performance=performance,
        active_alerts=0,
    )


def test_check_alerts_triggers_expected_notifications(monitoring_system: MonitoringSystem) -> None:
    recorded: list[dict[str, Any]] = []

    def fake_create_alert(level, source, message, title=None, context=None):  # noqa: ANN001
        recorded.append(
            {
                "level": level,
                "source": source,
                "message": message,
                "context": context or {},
            }
        )
        return object()

    monitoring_system.alert_manager.create_alert = fake_create_alert  # type: ignore[assignment]

    health = build_system_health(
        monitoring_system.config,
        cpu=80.0,
        memory=85.0,
        disk=90.0,
        error_rate=0.5,
        response_ms=1000.0,
    )
    monitoring_system._check_alerts(health)

    assert {entry["source"] for entry in recorded} == {"System", "Performance"}
    assert any("CPU" in entry["message"] for entry in recorded)
    assert any("error rate" in entry["message"].lower() for entry in recorded)


def test_collect_health_uses_collectors(monkeypatch: pytest.MonkeyPatch, monitoring_system: MonitoringSystem) -> None:
    expected_resources = make_resources(cpu=55.0)
    expected_performance = make_performance(avg_response_ms=120.0)
    expected_components = {"executor": make_component(ComponentStatus.HEALTHY)}

    monitoring_system.resource_collector = types.SimpleNamespace(collect=lambda: expected_resources)
    monitoring_system.performance_collector = types.SimpleNamespace(collect=lambda: expected_performance)
    monitoring_system.component_collector = types.SimpleNamespace(collect=lambda: expected_components)
    monitoring_system.alert_manager.get_active_alerts = lambda: [1, 2]  # type: ignore[assignment]

    health = monitoring_system._collect_health()

    assert health.resource_usage.cpu_percent == expected_resources.cpu_percent
    assert health.performance.avg_response_time_ms == expected_performance.avg_response_time_ms
    assert health.components == expected_components
    assert health.active_alerts == 2


def test_monitoring_loop_handles_exception_and_recovers(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    system = MonitoringSystem()
    caplog.set_level("ERROR")

    health = build_system_health(
        system.config,
        cpu=10.0,
        memory=10.0,
        disk=10.0,
        error_rate=0.0,
        response_ms=10.0,
    )

    call_counter = {"count": 0}

    def wrapped_collect() -> SystemHealth:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise RuntimeError("boom")
        return health

    monkeypatch.setattr(system, "_collect_health", wrapped_collect)
    monkeypatch.setattr(system.alert_manager, "get_active_alerts", lambda: [])
    monkeypatch.setattr(system, "_check_alerts", lambda _: None)

    def fake_sleep(_: float) -> None:
        if call_counter["count"] > 1:
            system.is_running = False

    monkeypatch.setattr("time.sleep", fake_sleep)

    system.is_running = True
    system._monitoring_loop()

    assert call_counter["count"] >= 2  # one failure + one success
    assert system.health_history[-1] is health
    assert any("Monitoring loop error" in record.message for record in caplog.records)


def test_start_and_stop_run_monitoring_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    system = MonitoringSystem()
    loop_calls: list[None] = []

    def fake_loop(self: MonitoringSystem) -> None:
        loop_calls.append(None)
        self.is_running = False

    monkeypatch.setattr(system, "_monitoring_loop", types.MethodType(fake_loop, system))

    system.start()
    system.thread.join(timeout=1)
    assert loop_calls

    system.stop()
    assert system.is_running is False


def test_start_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    system = MonitoringSystem()

    def fake_loop(self: MonitoringSystem) -> None:
        self.is_running = False

    monkeypatch.setattr(system, "_monitoring_loop", types.MethodType(fake_loop, system))

    starts: list[None] = []

    class DummyThread:
        def __init__(self, target) -> None:  # noqa: ANN001
            self._target = target

        def start(self) -> None:
            starts.append(None)
            self._target()

        def join(self, timeout: float | None = None) -> None:  # noqa: ARG002, ANN001
            return None

    monkeypatch.setattr("bot_v2.monitoring.system.engine.threading.Thread", lambda target: DummyThread(target))

    system.start()
    first_thread = system.thread
    assert len(starts) == 1

    system.is_running = True
    system.start()  # should no-op when already running
    assert system.thread is first_thread


def test_stop_handles_states(monkeypatch: pytest.MonkeyPatch) -> None:
    system = MonitoringSystem()
    system.stop()  # should return early when not running

    joined: list[float | None] = []

    class DummyThread:
        def join(self, timeout: float | None = None) -> None:
            joined.append(timeout)

    system.thread = DummyThread()
    system.is_running = True
    system.stop()
    assert joined == [5]
    assert system.is_running is False


def test_monitoring_loop_trims_history(monkeypatch: pytest.MonkeyPatch) -> None:
    system = MonitoringSystem()
    system.max_history = 1

    health1 = build_system_health(system.config, cpu=5.0, memory=5.0, disk=5.0, error_rate=0.0, response_ms=5.0)
    health2 = build_system_health(system.config, cpu=6.0, memory=6.0, disk=6.0, error_rate=0.0, response_ms=6.0)
    healths = [health1, health2]

    def wrapped_collect() -> SystemHealth:
        return healths.pop(0)

    monkeypatch.setattr(system, "_collect_health", wrapped_collect)
    monkeypatch.setattr(system, "_check_alerts", lambda _: None)

    def fake_sleep(_: float) -> None:
        if not healths:
            system.is_running = False

    monkeypatch.setattr("time.sleep", fake_sleep)

    system.is_running = True
    system._monitoring_loop()

    assert system.health_history == [health2]
