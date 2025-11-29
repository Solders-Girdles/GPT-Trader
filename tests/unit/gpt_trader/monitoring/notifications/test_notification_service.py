"""Unit tests for the notification service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications.backends import (
    ConsoleNotificationBackend,
    FileNotificationBackend,
)
from gpt_trader.monitoring.notifications.service import (
    NotificationService,
    create_notification_service,
)


class TestConsoleNotificationBackend:
    """Tests for ConsoleNotificationBackend."""

    def test_name_property(self) -> None:
        backend = ConsoleNotificationBackend()
        assert backend.name == "console"

    def test_is_enabled_property(self) -> None:
        backend = ConsoleNotificationBackend(enabled=True)
        assert backend.is_enabled is True

        backend = ConsoleNotificationBackend(enabled=False)
        assert backend.is_enabled is False

    @pytest.mark.asyncio
    async def test_send_prints_to_console(self, capsys) -> None:
        backend = ConsoleNotificationBackend(use_colors=False)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test message",
        )

        result = await backend.send(alert)

        assert result is True
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Test Alert" in captured.out
        assert "This is a test message" in captured.out

    @pytest.mark.asyncio
    async def test_send_filters_by_severity(self, capsys) -> None:
        backend = ConsoleNotificationBackend(min_severity=AlertSeverity.ERROR)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Low Severity Alert",
            message="Should be filtered",
        )

        result = await backend.send(alert)

        assert result is True  # Filtered but not failed
        captured = capsys.readouterr()
        assert captured.out == ""  # Nothing printed

    @pytest.mark.asyncio
    async def test_test_connection_always_true(self) -> None:
        backend = ConsoleNotificationBackend()
        result = await backend.test_connection()
        assert result is True


class TestFileNotificationBackend:
    """Tests for FileNotificationBackend."""

    def test_name_property(self) -> None:
        backend = FileNotificationBackend(file_path="/tmp/alerts.jsonl")
        assert backend.name == "file"

    @pytest.mark.asyncio
    async def test_send_writes_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            backend = FileNotificationBackend(file_path=file_path)
            alert = Alert(
                severity=AlertSeverity.ERROR,
                title="File Test",
                message="Testing file output",
            )

            result = await backend.send(alert)

            assert result is True
            content = Path(file_path).read_text()
            assert "File Test" in content
            assert "Testing file output" in content
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_checks_writability(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            backend = FileNotificationBackend(file_path=file_path)
            result = await backend.test_connection()
            assert result is True
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_fails_for_invalid_path(self) -> None:
        backend = FileNotificationBackend(file_path="/nonexistent/path/alerts.jsonl")
        result = await backend.test_connection()
        assert result is False


class TestNotificationService:
    """Tests for NotificationService."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        backend = Mock()
        backend.name = "mock"
        backend.is_enabled = True
        backend.send = AsyncMock(return_value=True)
        backend.test_connection = AsyncMock(return_value=True)
        return backend

    def test_add_backend(self, mock_backend) -> None:
        service = NotificationService()
        service.add_backend(mock_backend)
        assert len(service.backends) == 1
        assert service.backends[0].name == "mock"

    def test_remove_backend(self, mock_backend) -> None:
        service = NotificationService()
        service.add_backend(mock_backend)
        result = service.remove_backend("mock")
        assert result is True
        assert len(service.backends) == 0

    def test_remove_nonexistent_backend(self) -> None:
        service = NotificationService()
        result = service.remove_backend("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_notify_sends_to_backend(self, mock_backend) -> None:
        service = NotificationService()
        service.add_backend(mock_backend)

        result = await service.notify(
            title="Test",
            message="Test message",
            severity=AlertSeverity.WARNING,
        )

        assert result is True
        mock_backend.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_filters_by_severity(self, mock_backend) -> None:
        service = NotificationService(min_severity=AlertSeverity.ERROR)
        service.add_backend(mock_backend)

        result = await service.notify(
            title="Low Severity",
            message="Should be filtered",
            severity=AlertSeverity.WARNING,
        )

        assert result is True  # Filtered, not failed
        mock_backend.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_rate_limiting(self, mock_backend) -> None:
        service = NotificationService(rate_limit_per_minute=2)
        service.add_backend(mock_backend)

        # First two should succeed
        await service.notify(title="First", message="msg", force=True)
        await service.notify(title="Second", message="msg", force=True)

        # Reset internal state to test rate limiting
        service._sent_count_this_minute = 2

        # Third should be rate limited (without force)
        result = await service.notify(title="Third", message="msg")

        assert result is False

    @pytest.mark.asyncio
    async def test_notify_deduplication(self, mock_backend) -> None:
        service = NotificationService(dedup_window_seconds=60)
        service.add_backend(mock_backend)

        # First call
        await service.notify(title="Duplicate", message="msg", source="test")

        # Second call with same title/source (should be deduped)
        result = await service.notify(title="Duplicate", message="msg", source="test")

        assert result is True  # Deduped, not failed
        assert mock_backend.send.call_count == 1  # Only one call

    @pytest.mark.asyncio
    async def test_notify_force_bypasses_dedup(self, mock_backend) -> None:
        service = NotificationService(dedup_window_seconds=60)
        service.add_backend(mock_backend)

        await service.notify(title="Duplicate", message="msg", source="test")
        await service.notify(title="Duplicate", message="msg", source="test", force=True)

        assert mock_backend.send.call_count == 2

    @pytest.mark.asyncio
    async def test_notify_disabled_service(self, mock_backend) -> None:
        service = NotificationService(enabled=False)
        service.add_backend(mock_backend)

        result = await service.notify(title="Test", message="msg")

        assert result is False
        mock_backend.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_alert(self, mock_backend) -> None:
        service = NotificationService()
        service.add_backend(mock_backend)

        alert = Alert(
            severity=AlertSeverity.ERROR,
            title="Pre-built Alert",
            message="Test",
        )

        result = await service.notify_alert(alert)

        assert result is True
        mock_backend.send.assert_called_once()
        sent_alert = mock_backend.send.call_args[0][0]
        assert sent_alert.title == "Pre-built Alert"

    @pytest.mark.asyncio
    async def test_test_backends(self, mock_backend) -> None:
        service = NotificationService()
        service.add_backend(mock_backend)

        results = await service.test_backends()

        assert results == {"mock": True}
        mock_backend.test_connection.assert_called_once()

    def test_get_status(self, mock_backend) -> None:
        service = NotificationService(
            min_severity=AlertSeverity.WARNING,
            rate_limit_per_minute=30,
        )
        service.add_backend(mock_backend)

        status = service.get_status()

        assert status["enabled"] is True
        assert status["min_severity"] == "warning"
        assert status["rate_limit"] == 30
        assert len(status["backends"]) == 1
        assert status["backends"][0]["name"] == "mock"

    @pytest.mark.asyncio
    async def test_multiple_backends(self) -> None:
        backend1 = Mock()
        backend1.name = "backend1"
        backend1.is_enabled = True
        backend1.send = AsyncMock(return_value=True)

        backend2 = Mock()
        backend2.name = "backend2"
        backend2.is_enabled = True
        backend2.send = AsyncMock(return_value=True)

        service = NotificationService()
        service.add_backend(backend1)
        service.add_backend(backend2)

        await service.notify(title="Test", message="msg")

        backend1.send.assert_called_once()
        backend2.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_backend_failure(self) -> None:
        backend1 = Mock()
        backend1.name = "failing"
        backend1.is_enabled = True
        backend1.send = AsyncMock(return_value=False)

        backend2 = Mock()
        backend2.name = "working"
        backend2.is_enabled = True
        backend2.send = AsyncMock(return_value=True)

        service = NotificationService()
        service.add_backend(backend1)
        service.add_backend(backend2)

        result = await service.notify(title="Test", message="msg")

        assert result is True  # At least one succeeded


class TestCreateNotificationService:
    """Tests for the factory function."""

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
