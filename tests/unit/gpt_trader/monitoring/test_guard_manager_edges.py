"""Edge-case tests for RuntimeGuardManager."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import Alert, GuardConfig
from gpt_trader.monitoring.guards.manager import RuntimeGuardManager


class _GuardStub:
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


def test_guard_manager_skips_disabled_guards() -> None:
    manager = RuntimeGuardManager()
    guard = _GuardStub("disabled_guard", enabled=False)
    manager.add_guard(guard)

    with patch("gpt_trader.monitoring.guards.manager.record_counter") as mock_counter:
        alerts = manager.check_all({})

    assert alerts == []
    assert guard.check_calls == 0
    mock_counter.assert_not_called()


def test_guard_manager_catches_guard_exceptions_and_records() -> None:
    manager = RuntimeGuardManager()
    guard = _GuardStub("exploding_guard", raise_exc=True)
    manager.add_guard(guard)

    with patch("gpt_trader.monitoring.guards.manager.logger") as mock_logger:
        with patch("gpt_trader.monitoring.guards.manager.record_counter") as mock_counter:
            alerts = manager.check_all({"metric": 1})

    assert alerts == []
    assert guard.check_calls == 1
    mock_logger.error.assert_called_once()
    mock_counter.assert_called_once_with(
        "gpt_trader_guard_checks_total",
        labels={"guard": "exploding_guard", "result": "error"},
    )


def test_guard_manager_records_execution_metrics() -> None:
    manager = RuntimeGuardManager()
    alert = Alert(
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        guard_name="latency_guard",
        severity=AlertSeverity.WARNING,
        message="slow",
    )
    guard = _GuardStub("latency_guard", result=alert)
    manager.add_guard(guard)

    with patch("gpt_trader.monitoring.guards.manager.record_counter") as mock_counter:
        alerts = manager.check_all({"latency": 5})

    assert len(alerts) == 1
    mock_counter.assert_called_once_with(
        "gpt_trader_guard_checks_total",
        labels={"guard": "latency_guard", "result": "success"},
    )


def test_guard_manager_empty_guard_list_no_side_effects() -> None:
    manager = RuntimeGuardManager()
    handler = MagicMock()
    manager.add_alert_handler(handler)

    with patch("gpt_trader.monitoring.guards.manager.record_counter") as mock_counter:
        alerts = manager.check_all({"metric": 1})

    assert alerts == []
    mock_counter.assert_not_called()
    handler.assert_not_called()
