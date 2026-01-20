"""Edge-case tests for HeartbeatService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.heartbeat as heartbeat_module
from gpt_trader.config.constants import HEARTBEAT_HEALTH_MULTIPLIER
from gpt_trader.monitoring.heartbeat import HeartbeatService


@pytest.mark.asyncio
async def test_start_disabled_does_not_schedule_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HeartbeatService(enabled=False)

    mock_create_task = MagicMock()
    monkeypatch.setattr(heartbeat_module.asyncio, "create_task", mock_create_task)
    task = await service.start()

    assert task is None
    assert service._task is None
    assert service._running is False
    mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_stop_idempotent_clears_task() -> None:
    service = HeartbeatService()
    service._running = True
    service._task = MagicMock()
    service._task.done.return_value = True

    await service.stop()
    assert service._running is False
    assert service._task is None

    await service.stop()
    assert service._running is False
    assert service._task is None


def test_status_and_health_toggle_with_missed_heartbeat() -> None:
    service = HeartbeatService(interval_seconds=10)
    service._running = True
    service._last_heartbeat = 100.0

    healthy_time = 100.0 + (10 * HEARTBEAT_HEALTH_MULTIPLIER) - 0.1
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(heartbeat_module.time, "time", lambda: healthy_time)
        status = service.get_status()
        assert status["seconds_since_last"] == pytest.approx(healthy_time - 100.0)
        assert service.is_healthy is True

    unhealthy_time = 100.0 + (10 * HEARTBEAT_HEALTH_MULTIPLIER) + 0.1
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(heartbeat_module.time, "time", lambda: unhealthy_time)
        status = service.get_status()
        assert status["seconds_since_last"] == pytest.approx(unhealthy_time - 100.0)
        assert service.is_healthy is False


@pytest.mark.asyncio
async def test_heartbeat_count_increments_on_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HeartbeatService(event_store=MagicMock(), bot_id="test-bot")

    monkeypatch.setattr(heartbeat_module.time, "time", lambda: 123.0)
    await service._send_heartbeat()
    await service._send_heartbeat()

    assert service._heartbeat_count == 2
    assert service._last_heartbeat == 123.0
    assert service.event_store.store.call_count == 2
