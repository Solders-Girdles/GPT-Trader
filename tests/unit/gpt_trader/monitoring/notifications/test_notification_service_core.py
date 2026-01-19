"""Unit tests for the notification service."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications.service import NotificationService


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

        await service.notify(title="First", message="msg", force=True)
        await service.notify(title="Second", message="msg", force=True)

        service._sent_count_this_minute = 2

        result = await service.notify(title="Third", message="msg")

        assert result is False

    @pytest.mark.asyncio
    async def test_notify_deduplication(self, mock_backend) -> None:
        service = NotificationService(dedup_window_seconds=60)
        service.add_backend(mock_backend)

        await service.notify(title="Duplicate", message="msg", source="test")

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
