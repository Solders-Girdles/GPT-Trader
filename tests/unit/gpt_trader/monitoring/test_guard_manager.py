"""Edge-case tests for RuntimeGuardManager and alert handling."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.guards.manager as manager_module
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import Alert, GuardConfig
from gpt_trader.monitoring.guards.manager import RuntimeGuardManager


class _AlertGuardStub:
    def __init__(self, *, auto_shutdown: bool) -> None:
        self.config = GuardConfig(
            name="stub_guard",
            auto_shutdown=auto_shutdown,
            severity=AlertSeverity.CRITICAL,
        )


class _CheckGuardStub:
    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        result: Alert | None = None,
        raise_exc: bool = False,
    ) -> None:
        self.config = GuardConfig(name=name, enabled=enabled)
        self._result = result
        self._raise_exc = raise_exc
        self.check_calls = 0

    def check(self, context):
        self.check_calls += 1
        if self._raise_exc:
            raise RuntimeError("boom")
        return self._result


class TestGuardManagerAlertHandling:
    """Tests for alert handling edge cases."""

    def test_handle_alert_invokes_handlers_and_tolerates_exceptions(self) -> None:
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

        manager._handle_alert(alert, _AlertGuardStub(auto_shutdown=False))

        assert calls == ["stub_guard"]

    def test_handle_alert_auto_shutdown_only_when_enabled(self) -> None:
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

        manager._handle_alert(alert, _AlertGuardStub(auto_shutdown=False))
        assert shutdown_calls == []

        manager._handle_alert(alert, _AlertGuardStub(auto_shutdown=True))
        assert shutdown_calls == [True]

    def test_handle_alert_uses_severity_mapped_logger_method(self) -> None:
        manager = RuntimeGuardManager()
        alert = Alert(
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            guard_name="stub_guard",
            severity=AlertSeverity.WARNING,
            message="warn",
        )

        fake_logger = MagicMock()
        fake_logger.warning = MagicMock()

        original_logger = manager_module.logger
        manager_module.logger = fake_logger
        try:
            manager._handle_alert(alert, _AlertGuardStub(auto_shutdown=False))
        finally:
            manager_module.logger = original_logger

        fake_logger.warning.assert_called_once()


@pytest.fixture
def record_counter_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_counter = MagicMock()
    monkeypatch.setattr(manager_module, "record_counter", mock_counter)
    return mock_counter


@pytest.fixture
def guard_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(manager_module, "logger", mock_logger)
    return mock_logger


class TestGuardManagerEdgeCases:
    """Tests for RuntimeGuardManager edge cases."""

    def test_guard_manager_skips_disabled_guards(self, record_counter_mock: MagicMock) -> None:
        manager = RuntimeGuardManager()
        guard = _CheckGuardStub("disabled_guard", enabled=False)
        manager.add_guard(guard)

        alerts = manager.check_all({})

        assert alerts == []
        assert guard.check_calls == 0
        record_counter_mock.assert_not_called()

    def test_guard_manager_catches_guard_exceptions_and_records(
        self,
        guard_logger: MagicMock,
        record_counter_mock: MagicMock,
    ) -> None:
        manager = RuntimeGuardManager()
        guard = _CheckGuardStub("exploding_guard", raise_exc=True)
        manager.add_guard(guard)

        alerts = manager.check_all({"metric": 1})

        assert alerts == []
        assert guard.check_calls == 1
        guard_logger.error.assert_called_once()
        record_counter_mock.assert_called_once_with(
            "gpt_trader_guard_checks_total",
            labels={"guard": "exploding_guard", "result": "error"},
        )

    def test_guard_manager_records_execution_metrics(
        self,
        record_counter_mock: MagicMock,
    ) -> None:
        manager = RuntimeGuardManager()
        alert = Alert(
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            guard_name="latency_guard",
            severity=AlertSeverity.WARNING,
            message="slow",
        )
        guard = _CheckGuardStub("latency_guard", result=alert)
        manager.add_guard(guard)

        alerts = manager.check_all({"latency": 5})

        assert len(alerts) == 1
        record_counter_mock.assert_called_once_with(
            "gpt_trader_guard_checks_total",
            labels={"guard": "latency_guard", "result": "success"},
        )

    def test_guard_manager_empty_guard_list_no_side_effects(
        self,
        record_counter_mock: MagicMock,
    ) -> None:
        manager = RuntimeGuardManager()
        handler = MagicMock()
        manager.add_alert_handler(handler)

        alerts = manager.check_all({"metric": 1})

        assert alerts == []
        record_counter_mock.assert_not_called()
        handler.assert_not_called()
