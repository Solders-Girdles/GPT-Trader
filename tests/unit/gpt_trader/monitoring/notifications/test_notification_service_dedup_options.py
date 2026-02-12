"""Dedup option behavior tests for the notification service."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.monitoring.notifications.service import NotificationService


@pytest.fixture
def mock_backend() -> Mock:
    backend = Mock()
    backend.name = "mock"
    backend.is_enabled = True
    backend.send = AsyncMock(return_value=True)
    backend.test_connection = AsyncMock(return_value=True)
    return backend


@pytest.mark.asyncio
async def test_notify_distinguishes_by_message_when_opted_in(mock_backend) -> None:
    service = NotificationService(
        dedup_window_seconds=60,
        dedup_include_message=True,
    )
    service.add_backend(mock_backend)

    await service.notify(title="Duplicate", message="first", source="test")
    await service.notify(title="Duplicate", message="second", source="test")

    assert mock_backend.send.call_count == 2


@pytest.mark.asyncio
async def test_notify_ignores_volatile_message_context_and_metadata_by_default(
    mock_backend,
) -> None:
    service = NotificationService(dedup_window_seconds=60)
    service.add_backend(mock_backend)

    await service.notify(
        title="Duplicate",
        message="stale_age_seconds=31 reconnect_count=2",
        source="test",
        context={"stale_age_seconds": 31, "reconnect_count": 2},
        metadata={"event_id": "evt-1"},
    )
    await service.notify(
        title="Duplicate",
        message="stale_age_seconds=44 reconnect_count=3",
        source="test",
        context={"stale_age_seconds": 44, "reconnect_count": 3},
        metadata={"event_id": "evt-2"},
    )

    assert mock_backend.send.call_count == 1


@pytest.mark.asyncio
async def test_notify_distinguishes_by_context_when_opted_in(mock_backend) -> None:
    service = NotificationService(
        dedup_window_seconds=60,
        dedup_include_context=True,
    )
    service.add_backend(mock_backend)

    await service.notify(
        title="Duplicate",
        message="same",
        source="test",
        context={"step": 1},
    )
    await service.notify(
        title="Duplicate",
        message="same",
        source="test",
        context={"step": 2},
    )

    assert mock_backend.send.call_count == 2
