"""Edge-case tests for NotificationService."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.monitoring.notifications.service import NotificationService


@pytest.mark.asyncio
async def test_notify_returns_false_without_backends() -> None:
    service = NotificationService()

    result = await service.notify(title="Test", message="No backends")

    assert result is False


@pytest.mark.asyncio
async def test_notify_returns_false_when_backends_disabled() -> None:
    backend = Mock()
    backend.name = "disabled"
    backend.is_enabled = False
    backend.send = AsyncMock(return_value=True)

    service = NotificationService()
    service.add_backend(backend)

    result = await service.notify(title="Test", message="Disabled backend")

    assert result is False
    backend.send.assert_not_called()


@pytest.mark.asyncio
async def test_notify_swallow_send_exception_single_backend() -> None:
    backend = Mock()
    backend.name = "failing"
    backend.is_enabled = True
    backend.send = AsyncMock(side_effect=RuntimeError("boom"))

    service = NotificationService()
    service.add_backend(backend)

    result = await service.notify(title="Test", message="Failure")

    assert result is False
    backend.send.assert_called_once()


@pytest.mark.asyncio
async def test_notify_rate_limit_resets_after_minute() -> None:
    backend = Mock()
    backend.name = "enabled"
    backend.is_enabled = True
    backend.send = AsyncMock(return_value=True)

    service = NotificationService(rate_limit_per_minute=1)
    service.add_backend(backend)
    service._sent_count_this_minute = service.rate_limit_per_minute
    service._minute_reset_time = datetime.utcnow() - timedelta(minutes=2)

    result = await service.notify(title="Test", message="Rate reset")

    assert result is True
    backend.send.assert_called_once()


def test_cleanup_old_alerts_prunes_expired_entries() -> None:
    service = NotificationService(dedup_window_seconds=10)
    now = datetime.utcnow()

    service._recent_alerts = {
        "expired": now - timedelta(seconds=21),
        "fresh": now - timedelta(seconds=19),
    }

    service._cleanup_old_alerts()

    assert "expired" not in service._recent_alerts
    assert "fresh" in service._recent_alerts
