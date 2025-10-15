from __future__ import annotations

from bot_v2.monitoring.alert_types import AlertSeverity
from bot_v2.monitoring.interfaces import ComponentStatus
from bot_v2.monitoring.system.engine import MonitoringSystem


def _build_system(collectors, alert_recorder) -> MonitoringSystem:
    system = MonitoringSystem(
        resource_collector=collectors["resource"],
        performance_collector=collectors["performance"],
        component_collector=collectors["component"],
    )
    alert_recorder.attach(system.alert_manager)
    return system


def _run_single_iteration(system: MonitoringSystem) -> None:
    health = system._collect_health()
    system.health_history.append(health)
    if len(system.health_history) > system.max_history:
        system.health_history.pop(0)
    system._check_alerts(health)


def test_monitoring_system_generates_alerts(monitoring_collectors, alert_recorder, frozen_time):
    collectors = monitoring_collectors
    collectors["resource"].update(cpu_percent=95.0, memory_percent=92.0, disk_percent=91.0)
    collectors["performance"].update(error_rate=0.2, avg_response_time_ms=250.0)
    collectors["component"].set_component(
        "RiskEngine",
        status=ComponentStatus.UNHEALTHY,
        error_count=3,
    )

    system = _build_system(collectors, alert_recorder)
    _run_single_iteration(system)

    assert len(system.health_history) == 1
    health = system.get_current_health()
    assert health is not None
    assert health.overall_status in {ComponentStatus.UNHEALTHY, ComponentStatus.DEGRADED}

    severities = {alert.severity for alert in alert_recorder.alerts}
    assert AlertSeverity.CRITICAL in severities
    assert AlertSeverity.ERROR in severities
    assert AlertSeverity.WARNING in severities


def test_monitoring_system_history_trimming(monitoring_collectors, alert_recorder):
    system = _build_system(monitoring_collectors, alert_recorder)
    system.max_history = 3

    for _ in range(5):
        _run_single_iteration(system)

    assert len(system.health_history) == 3


def test_monitoring_system_start_stop(monkeypatch, monitoring_collectors, alert_recorder):
    system = _build_system(monitoring_collectors, alert_recorder)

    monkeypatch.setattr("builtins.print", lambda *_, **__: None)

    class DummyThread:
        def __init__(self, target):
            self._target = target
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self._target()

    monkeypatch.setattr("bot_v2.monitoring.system.engine.threading.Thread", DummyThread)

    system.start()
    assert system.is_running
    assert isinstance(system.thread, DummyThread)
    assert system.thread.started

    system.stop()
    assert not system.is_running


def test_monitoring_system_accessors_when_idle(monitoring_collectors, alert_recorder):
    system = _build_system(monitoring_collectors, alert_recorder)
    assert system.get_current_health() is None
    assert system.get_alerts() == []
    assert system.get_active_alerts() == []
