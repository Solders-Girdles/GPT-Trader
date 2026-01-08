from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from gpt_trader.features.live_trade import guard_errors
from gpt_trader.monitoring import metrics_collector
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


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics collector between tests."""
    metrics_collector.reset_all()
    yield
    metrics_collector.reset_all()


@pytest.fixture
def counter_spy(monkeypatch):
    """Spy on the real metrics_record_counter function."""
    calls: list[tuple[str, dict[str, str] | None]] = []

    original_record_counter = metrics_collector.record_counter

    def _record_counter(
        name: str, increment: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        calls.append((name, labels))
        # Also call original for proper metrics collection
        original_record_counter(name, increment, labels)

    monkeypatch.setattr(guard_errors, "_metrics_record_counter", _record_counter)
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

    # Check Prometheus-style counter with labels
    assert counter_spy == [
        (
            "gpt_trader_guard_trips_total",
            {"guard": "latency_guard", "category": "telemetry", "recoverable": "true"},
        ),
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

    # Check Prometheus-style counter with labels
    assert counter_spy == [
        (
            "gpt_trader_guard_trips_total",
            {"guard": "exposure", "category": "action", "recoverable": "false"},
        ),
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

    # Check Prometheus-style counter with labels
    assert counter_spy == [
        (
            "gpt_trader_guard_checks_total",
            {"guard": "latency_guard", "result": "success"},
        ),
    ]


def test_guard_metrics_stored_in_collector() -> None:
    """Test that guard metrics are actually stored in the metrics collector."""
    # Record a failure
    error = guard_errors.RiskLimitExceeded(
        "Position Size", "Exceeded max position", details={"max": 100, "requested": 150}
    )
    guard_errors.record_guard_failure(error)

    # Record a success
    guard_errors.record_guard_success("Leverage Check")

    # Check metrics summary
    summary = metrics_collector.get_metrics_collector().get_metrics_summary()

    # Verify counters include our metrics
    counters = summary["counters"]

    # Check guard trip counter
    trip_key = "gpt_trader_guard_trips_total{category=limit,guard=position_size,recoverable=false}"
    assert trip_key in counters
    assert counters[trip_key] == 1

    # Check guard success counter
    success_key = "gpt_trader_guard_checks_total{guard=leverage_check,result=success}"
    assert success_key in counters
    assert counters[success_key] == 1
