"""Edge-case tests for RuntimeGuardManager alert handling."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import Alert, GuardConfig
from gpt_trader.monitoring.guards.manager import RuntimeGuardManager


class _GuardStub:
    def __init__(self, *, auto_shutdown: bool) -> None:
        self.config = GuardConfig(
            name="stub_guard",
            auto_shutdown=auto_shutdown,
            severity=AlertSeverity.CRITICAL,
        )


def test_handle_alert_invokes_handlers_and_tolerates_exceptions() -> None:
    manager = RuntimeGuardManager()

    calls = []

    def good_handler(alert: Alert) -> None:
        calls.append(alert.guard_name)

    def bad_handler(alert: Alert) -> None:
        raise RuntimeError("handler failed")

    manager.add_alert_handler(bad_handler)
    manager.add_alert_handler(good_handler)

    alert = Alert(
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        guard_name="stub_guard",
        severity=AlertSeverity.ERROR,
        message="boom",
    )

    manager._handle_alert(alert, _GuardStub(auto_shutdown=False))

    assert calls == ["stub_guard"]


def test_handle_alert_auto_shutdown_only_when_enabled() -> None:
    manager = RuntimeGuardManager()
    shutdown_calls = []

    def shutdown_callback() -> None:
        shutdown_calls.append(True)

    manager.set_shutdown_callback(shutdown_callback)

    alert = Alert(
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        guard_name="stub_guard",
        severity=AlertSeverity.CRITICAL,
        message="critical",
    )

    manager._handle_alert(alert, _GuardStub(auto_shutdown=False))
    assert shutdown_calls == []

    manager._handle_alert(alert, _GuardStub(auto_shutdown=True))
    assert shutdown_calls == [True]


def test_handle_alert_uses_severity_mapped_logger_method() -> None:
    manager = RuntimeGuardManager()
    alert = Alert(
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        guard_name="stub_guard",
        severity=AlertSeverity.WARNING,
        message="warn",
    )

    fake_logger = MagicMock()
    fake_logger.warning = MagicMock()

    from gpt_trader.monitoring.guards import manager as manager_module

    original_logger = manager_module.logger
    manager_module.logger = fake_logger
    try:
        manager._handle_alert(alert, _GuardStub(auto_shutdown=False))
    finally:
        manager_module.logger = original_logger

    fake_logger.warning.assert_called_once()
