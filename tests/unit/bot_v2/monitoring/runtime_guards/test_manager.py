from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from bot_v2.monitoring.alerts import AlertSeverity
from bot_v2.monitoring.runtime_guards.base import Alert, GuardConfig, GuardStatus, RuntimeGuard
from bot_v2.monitoring.runtime_guards.manager import RuntimeGuardManager, create_default_guards


class StubGuard(RuntimeGuard):
    def __init__(
        self, name: str, alert: Alert | None = None, *, auto_shutdown: bool = False
    ) -> None:
        super().__init__(
            GuardConfig(
                name=name,
                threshold=1.0,
                severity=AlertSeverity.ERROR,
                auto_shutdown=auto_shutdown,
            )
        )
        self._alert = alert
        self.reset_called = False

    def check(self, context: dict[str, Any]) -> Alert | None:  # type: ignore[override]
        self.last_check = datetime.now()
        if self._alert:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1
            self.last_alert = datetime.now()
        return self._alert

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self.reset_called = True


def make_alert(name: str) -> Alert:
    return Alert(
        timestamp=datetime.now(),
        guard_name=name,
        severity=AlertSeverity.CRITICAL,
        message="breached",
        context={"value": 42},
    )


def test_manager_handles_alert_handlers_and_shutdown(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR")
    manager = RuntimeGuardManager()
    triggered: list[Alert] = []
    shutdown_called = {"flag": False}

    alert = make_alert("daily_loss")
    guard = StubGuard("daily_loss", alert=alert, auto_shutdown=True)
    manager.add_guard(guard)

    def good_handler(item: Alert) -> None:
        triggered.append(item)

    def bad_handler(item: Alert) -> None:  # noqa: ARG001
        raise RuntimeError("handler failure")

    manager.add_alert_handler(good_handler)
    manager.add_alert_handler(bad_handler)
    manager.set_shutdown_callback(lambda: shutdown_called.update(flag=True))

    alerts = manager.check_all({"value": 100})

    assert alerts == [alert]
    assert triggered == [alert]
    assert shutdown_called["flag"] is True
    assert any("Alert handler error" in record.message for record in caplog.records)


def test_manager_status_and_reset() -> None:
    manager = RuntimeGuardManager()
    guard = StubGuard("error_rate")
    manager.add_guard(guard)

    status = manager.get_status()
    assert status["error_rate"]["enabled"] is True

    manager.reset_guard("error_rate")
    assert guard.reset_called is True

    guard.reset_called = False
    manager.reset_all()
    assert guard.reset_called is True


def test_create_default_guards_handles_invalid_config() -> None:
    config = {
        "risk_management": "not-a-dict",
    }
    manager = create_default_guards(config)

    # Default guards should still be created even if config types are wrong
    guard_names = set(manager.guards.keys())
    assert {
        "daily_loss",
        "stale_marks",
        "error_rate",
        "position_stuck",
        "max_drawdown",
    } <= guard_names


def test_create_default_guards_uses_threshold_overrides() -> None:
    config = {
        "risk_management": {
            "daily_loss_limit": "250.5",
            "max_drawdown_pct": "12.5",
            "circuit_breakers": {
                "stale_mark_seconds": "15",
                "error_threshold": "3",
            },
        }
    }
    manager = create_default_guards(config)
    daily_loss = manager.guards["daily_loss"]
    assert daily_loss.config.threshold == 250.5
    drawdown = manager.guards["max_drawdown"]
    assert drawdown.config.threshold == 12.5


def test_get_status_reflects_breach_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RuntimeGuardManager()
    alert = make_alert("drawdown")
    guard = StubGuard("drawdown", alert=alert)
    manager.add_guard(guard)

    manager.check_all({})
    status = manager.get_status()["drawdown"]
    assert status["status"] == "breached"
    assert status["breach_count"] == 1
    assert status["last_alert"].startswith(str(alert.timestamp.date()))

    manager.reset_guard("drawdown")
    status = manager.get_status()["drawdown"]
    assert status["status"] == "healthy"
