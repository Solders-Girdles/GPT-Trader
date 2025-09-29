"""Unit tests for the baseline RuntimeGuard functionality."""

from __future__ import annotations

import pytest

from bot_v2.monitoring.runtime_guards import (
    AlertSeverity,
    GuardConfig,
    GuardStatus,
    RuntimeGuard,
)


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
