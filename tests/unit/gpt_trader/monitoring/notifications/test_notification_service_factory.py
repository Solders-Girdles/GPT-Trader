"""Tests for the notification service factory."""

from __future__ import annotations

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.notifications.service import create_notification_service


class TestCreateNotificationService:
    def test_creates_with_console_only(self) -> None:
        service = create_notification_service(console_enabled=True)
        assert len(service.backends) == 1
        assert service.backends[0].name == "console"

    def test_creates_with_webhook(self) -> None:
        service = create_notification_service(
            console_enabled=True,
            webhook_url="https://hooks.slack.com/test",
        )
        assert len(service.backends) == 2
        backend_names = [b.name for b in service.backends]
        assert "console" in backend_names
        assert "webhook" in backend_names

    def test_creates_with_file(self) -> None:
        service = create_notification_service(
            console_enabled=True,
            file_path="/tmp/alerts.jsonl",
        )
        assert len(service.backends) == 2
        backend_names = [b.name for b in service.backends]
        assert "console" in backend_names
        assert "file" in backend_names

    def test_respects_min_severity(self) -> None:
        service = create_notification_service(
            min_severity=AlertSeverity.CRITICAL,
        )
        assert service.min_severity == AlertSeverity.CRITICAL
