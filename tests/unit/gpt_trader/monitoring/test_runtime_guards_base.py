"""Unit tests for the baseline RuntimeGuard functionality."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards import (
    GuardConfig,
    GuardStatus,
    RuntimeGuard,
)


class ManualTimeProvider:
    def __init__(self, start: datetime) -> None:
        self._current = start
        self._monotonic = start.timestamp()

    def now_utc(self) -> datetime:
        return self._current

    def time(self) -> float:
        return self._current.timestamp()

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, delta: timedelta) -> None:
        self._current = self._current + delta
        self._monotonic += delta.total_seconds()


def test_runtime_guard_triggers_generic_breach():
    guard = RuntimeGuard(GuardConfig(name="latency", threshold=100.0, severity=AlertSeverity.ERROR))

    context = {"value": 125.4, "units": "ms"}
    alert = guard.check(context)

    assert alert is not None
    assert guard.status is GuardStatus.BREACHED
    assert "latency" in alert.message.lower()
    assert "125.4" in alert.message
    assert alert.context == context


def test_runtime_guard_warning_before_breach():
    guard = RuntimeGuard(GuardConfig(name="cpu", threshold=90.0, severity=AlertSeverity.WARNING))

    # First pass should mark as warning but not raise alert
    context = {"value": 70, "warning_ratio": 0.75, "units": "%"}
    alert = guard.check(context)

    assert alert is None
    assert guard.status is GuardStatus.WARNING

    # Exceeding the threshold should now breach
    alert = guard.check({"value": 95, "units": "%"})
    assert alert is not None
    assert guard.status is GuardStatus.BREACHED


def test_runtime_guard_less_than_comparison():
    guard = RuntimeGuard(
        GuardConfig(name="heartbeat_delay", threshold=5.0, severity=AlertSeverity.WARNING)
    )

    context = {"value": 2.5, "comparison": "lt", "units": "s"}
    alert = guard.check(context)

    assert alert is not None
    assert "dropped below" in alert.message


def test_runtime_guard_absolute_comparison():
    guard = RuntimeGuard(GuardConfig(name="balance", threshold=10.0, severity=AlertSeverity.ERROR))

    context = {"value": -12, "comparison": "abs_gt", "label": "Account delta"}
    alert = guard.check(context)

    assert alert is not None
    assert "absolute" in alert.message.lower()


def test_runtime_guard_uses_time_provider_for_cooldown():
    provider = ManualTimeProvider(datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC))
    guard = RuntimeGuard(
        GuardConfig(
            name="latency",
            threshold=100.0,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=60,
        ),
        time_provider=provider,
    )

    first_alert = guard.check({"value": 150, "units": "ms"})
    assert first_alert is not None
    assert first_alert.timestamp == provider.now_utc()
    assert guard.last_alert == provider.now_utc()

    provider.advance(timedelta(seconds=30))
    assert guard.check({"value": 150, "units": "ms"}) is None

    provider.advance(timedelta(seconds=31))
    second_alert = guard.check({"value": 150, "units": "ms"})
    assert second_alert is not None
    assert second_alert.timestamp == provider.now_utc()


def test_runtime_guard_cooldown_is_per_metric_key():
    provider = ManualTimeProvider(datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC))
    guard = RuntimeGuard(
        GuardConfig(
            name="latency",
            threshold=100.0,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=60,
        ),
        time_provider=provider,
    )

    first_alert = guard.check({"metric_key": "latency_p95", "latency_p95": 150, "units": "ms"})
    assert first_alert is not None

    provider.advance(timedelta(seconds=30))
    assert guard.check({"metric_key": "latency_p95", "latency_p95": 150, "units": "ms"}) is None

    second_alert = guard.check({"metric_key": "latency_p99", "latency_p99": 200, "units": "ms"})
    assert second_alert is not None


def test_runtime_guard_cooldown_is_per_explicit_cooldown_key():
    provider = ManualTimeProvider(datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC))
    guard = RuntimeGuard(
        GuardConfig(
            name="launcher_starvation",
            threshold=1.0,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=60,
        ),
        time_provider=provider,
    )

    first_alert = guard.check({"value": 2, "cooldown_key": "opp_a"})
    assert first_alert is not None

    provider.advance(timedelta(seconds=20))
    second_alert = guard.check({"value": 2, "cooldown_key": "opp_b"})
    assert second_alert is not None

    provider.advance(timedelta(seconds=20))
    assert guard.check({"value": 2, "cooldown_key": "opp_a"}) is None

    provider.advance(timedelta(seconds=21))
    third_alert = guard.check({"value": 2, "cooldown_key": "opp_a"})
    assert third_alert is not None
