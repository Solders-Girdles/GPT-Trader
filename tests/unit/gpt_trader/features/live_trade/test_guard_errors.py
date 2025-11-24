from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from gpt_trader.features.live_trade import guard_errors
from gpt_trader.monitoring.alert_types import AlertSeverity


@pytest.fixture(autouse=True)
def reset_alert_system():
    """Ensure guard alert system state is isolated between tests."""
    original = guard_errors._alert_system
    guard_errors.configure_guard_alert_system(None)
    try:
        yield
    finally:
        guard_errors._alert_system = original


@pytest.fixture
def counter_spy(monkeypatch):
    calls: list[tuple[str, int]] = []

    def _record_counter(name: str, increment: int = 1) -> None:
        calls.append((name, increment))

    monkeypatch.setattr(guard_errors, "record_counter", _record_counter)
    return calls


@dataclass
class AlertSpy:
    calls: list[tuple[AlertSeverity, str, str, dict]] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def trigger_alert(
        self, severity: AlertSeverity, category: str, message: str, metadata: dict | None = None
    ) -> None:
        self.calls.append((severity, category, message, metadata or {}))


def test_configure_guard_alert_system_accepts_custom_dispatcher():
    custom = AlertSpy()
    guard_errors.configure_guard_alert_system(custom)

    assert guard_errors._get_alert_system() is custom


def test_get_alert_system_reinitializes_invalid_object():
    guard_errors.configure_guard_alert_system(object())

    system = guard_errors._get_alert_system()

    assert isinstance(system, guard_errors.GuardAlertDispatcher)
    assert hasattr(system, "trigger_alert")


def test_record_guard_failure_recoverable_emits_warning(
    monkeypatch: pytest.MonkeyPatch, counter_spy, caplog: pytest.LogCaptureFixture
) -> None:
    alert_spy = AlertSpy()
    monkeypatch.setattr(guard_errors, "_get_alert_system", lambda: alert_spy)

    caplog.set_level(logging.WARNING, guard_errors.logger.name)

    error = guard_errors.RiskGuardTelemetryError(
        "Latency Guard", "Rate limited", details={"attempts": 3}
    )
    guard_errors.record_guard_failure(error)

    assert counter_spy == [
        ("risk.guards.latency_guard.recoverable_failures", 1),
        ("risk.guards.latency_guard.telemetry", 1),
    ]
    assert alert_spy.calls == [
        (
            AlertSeverity.WARNING,
            "risk_guard.Latency Guard",
            "Rate limited",
            {"attempts": 3},
        )
    ]

    records = [rec for rec in caplog.records if rec.message == "Rate limited"]
    assert records and records[0].levelno == logging.WARNING
    assert records[0].guard_failure == {
        "guard": "Latency Guard",
        "category": "telemetry",
        "recoverable": True,
        "details": {"attempts": 3},
    }


def test_record_guard_failure_critical_emits_error(
    monkeypatch: pytest.MonkeyPatch, counter_spy, caplog: pytest.LogCaptureFixture
) -> None:
    alert_spy = AlertSpy()
    monkeypatch.setattr(guard_errors, "_get_alert_system", lambda: alert_spy)

    caplog.set_level(logging.ERROR, guard_errors.logger.name)

    error = guard_errors.RiskGuardActionError(
        "Exposure", "Threshold breached", details={"breach_amount": 42_000}
    )
    guard_errors.record_guard_failure(error)

    assert counter_spy == [
        ("risk.guards.exposure.critical_failures", 1),
        ("risk.guards.exposure.action", 1),
    ]
    assert alert_spy.calls == [
        (
            AlertSeverity.CRITICAL,
            "risk_guard.Exposure",
            "Threshold breached",
            {"breach_amount": 42_000},
        )
    ]

    records = [rec for rec in caplog.records if rec.message == "Threshold breached"]
    assert records and records[0].levelno == logging.ERROR
    assert records[0].guard_failure == {
        "guard": "Exposure",
        "category": "action",
        "recoverable": False,
        "details": {"breach_amount": 42_000},
    }


def test_record_guard_success_increments_counter(counter_spy) -> None:
    guard_errors.record_guard_success("Latency Guard")

    assert counter_spy == [("risk.guards.latency_guard.success", 1)]
