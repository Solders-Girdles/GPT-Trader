from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bot_v2.monitoring.alerts import AlertDispatcher, AlertLevel
from bot_v2.monitoring.alerts_manager import AlertManager


class FrozenDateTime(datetime):
    current = datetime(2025, 1, 15, 12, 0, 0)

    @classmethod
    def now(cls) -> datetime:
        return cls.current

    @classmethod
    def utcnow(cls) -> datetime:
        return cls.current

    @classmethod
    def advance(cls, **delta: int) -> None:
        cls.current += timedelta(**delta)

    @classmethod
    def move_to(cls, new_time: datetime) -> None:
        cls.current = new_time


@pytest.fixture(autouse=True)
def patch_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot_v2.monitoring.alerts_manager.datetime", FrozenDateTime)


def make_manager() -> AlertManager:
    dispatcher = AlertDispatcher.from_config({})
    return AlertManager(dispatcher=dispatcher, dedup_window_seconds=60)


def test_create_alert_handles_dedupe_and_clock_regressions() -> None:
    manager = make_manager()
    FrozenDateTime.move_to(datetime(2025, 1, 15, 12, 0, 0))

    first = manager.create_alert(AlertLevel.INFO, "System", "Spike")
    assert first is not None

    FrozenDateTime.advance(seconds=30)
    second = manager.create_alert(AlertLevel.INFO, "System", "Spike")
    assert second is None  # deduped within window

    FrozenDateTime.advance(seconds=90)
    third = manager.create_alert(AlertLevel.INFO, "System", "Spike")
    assert third is not None

    # Move the clock backwards – dedupe entry should be cleared instead of blocking
    FrozenDateTime.advance(minutes=-10)
    rebound = manager.create_alert(AlertLevel.INFO, "System", "Spike")
    assert rebound is not None


def test_cleanup_old_alerts_purges_history_and_recent() -> None:
    manager = make_manager()

    start = datetime(2025, 1, 15, 12, 0, 0)
    FrozenDateTime.move_to(start - timedelta(hours=30))
    manager.create_alert(AlertLevel.INFO, "Test", "Old")

    FrozenDateTime.move_to(start - timedelta(hours=2))
    manager.create_alert(AlertLevel.INFO, "Test", "Recent")

    FrozenDateTime.move_to(start + timedelta(hours=30))
    manager.create_alert(AlertLevel.INFO, "Test", "Future")

    manager._recent_alerts["Test:Old"] = start - timedelta(hours=30)
    manager._recent_alerts["Test:Future"] = start + timedelta(hours=40)

    FrozenDateTime.move_to(start + timedelta(hours=20))
    manager.cleanup_old_alerts(retention_hours=24)

    messages = [alert.message for alert in manager.alert_history]
    assert messages == ["Recent"]
    assert "Test:Old" not in manager._recent_alerts
    assert "Test:Future" not in manager._recent_alerts


def test_recent_alerts_and_summary_filters(tmp_path: Path) -> None:
    manager = make_manager()
    FrozenDateTime.move_to(datetime(2025, 1, 15, 14, 0, 0))
    manager.create_alert("warning", "System", "Warn alert")

    FrozenDateTime.advance(minutes=10)
    manager.create_alert(AlertLevel.ERROR, "Trading", "Error alert")

    recent_errors = manager.get_recent_alerts(severity=AlertLevel.ERROR)
    assert len(recent_errors) == 1 and recent_errors[0].source == "Trading"

    recent_system = manager.get_recent_alerts(source="System")
    assert len(recent_system) == 1 and recent_system[0].message == "Warn alert"

    summary = manager.get_alert_summary()
    assert summary["total"] == 2
    assert summary["by_severity"]["warning"] == 1
    assert "System" in summary["sources"]

    FrozenDateTime.advance(minutes=50)
    active = manager.get_active_alerts()
    assert [alert.message for alert in active] == ["Error alert"]

    FrozenDateTime.advance(hours=2)
    inactive = manager.get_active_alerts()
    assert inactive == []

    # Verify history trim honouring max_history
    manager.max_history = 2
    manager.create_alert(AlertLevel.INFO, "System", "Info 1")
    manager.create_alert(AlertLevel.INFO, "System", "Info 2")
    messages = [alert.message for alert in manager.alert_history]
    assert messages == ["Info 1", "Info 2"]

    # Verify YAML mapping via temporary file
    yaml_content = {
        "monitoring": {
            "alerts": {
                "channels": [
                    {"type": "slack", "webhook_url": "https://example", "level": "error"},
                    {"type": "pagerduty", "api_key": "pd-key", "level": "critical"},
                ]
            }
        }
    }
    config_file = tmp_path / "alerts.yaml"
    config_file.write_text(
        "monitoring:\n  alerts:\n    channels:\n      - type: slack\n        webhook_url: https://example\n        level: error\n      - type: pagerduty\n        api_key: pd-key\n        level: critical\n"
    )
    manager_from_yaml = AlertManager.from_profile_yaml(config_file)
    assert isinstance(manager_from_yaml.dispatcher, AlertDispatcher)
    assert manager_from_yaml.dispatcher.channels.get("slack").min_severity == AlertLevel.ERROR
