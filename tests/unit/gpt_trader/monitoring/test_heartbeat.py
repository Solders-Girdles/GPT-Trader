"""Unit tests for the heartbeat service."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest

import gpt_trader.monitoring.heartbeat as heartbeat
from gpt_trader.monitoring.heartbeat import EVENT_HEARTBEAT, HeartbeatService


def _freeze_time(
    monkeypatch: pytest.MonkeyPatch,
    start: float = 1000.0,
) -> Callable[[float], None]:
    now = {"value": start}
    monkeypatch.setattr(heartbeat.time, "time", lambda: now["value"])

    def advance(seconds: float) -> None:
        now["value"] += seconds

    return advance


class TestHeartbeatServiceInit:
    def test_default_values(self) -> None:
        service = HeartbeatService()
        assert service.event_store is None
        assert service.ping_url is None
        assert service.interval_seconds == 60
        assert service.bot_id == ""
        assert service.enabled is True
        assert service._running is False
        assert service._task is None
        assert service._heartbeat_count == 0

    def test_custom_values(self) -> None:
        mock_store = Mock()
        service = HeartbeatService(
            event_store=mock_store,
            ping_url="https://hc-ping.com/test",
            interval_seconds=30,
            bot_id="test-bot",
            enabled=False,
        )
        assert service.event_store is mock_store
        assert service.ping_url == "https://hc-ping.com/test"
        assert service.interval_seconds == 30
        assert service.bot_id == "test-bot"
        assert service.enabled is False


class TestHeartbeatServiceStart:
    @pytest.mark.asyncio
    async def test_start_when_disabled(self) -> None:
        service = HeartbeatService(enabled=False)
        task = await service.start()
        assert task is None
        assert service._running is False

    @pytest.mark.asyncio
    async def test_start_creates_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=1)

        async def no_op_loop() -> None:
            return None

        monkeypatch.setattr(service, "_heartbeat_loop", no_op_loop)

        task = await service.start()
        try:
            assert task is not None
            assert service._running is True
            assert service._task is task
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=1)

        async def no_op_loop() -> None:
            return None

        monkeypatch.setattr(service, "_heartbeat_loop", no_op_loop)

        task1 = await service.start()
        task2 = await service.start()

        try:
            assert task1 is task2
        finally:
            await service.stop()


class TestHeartbeatServiceStop:
    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        service = HeartbeatService()
        await service.stop()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        service = HeartbeatService(interval_seconds=10)

        event = asyncio.Event()

        async def wait_forever() -> None:
            await event.wait()

        service._running = True
        service._task = asyncio.create_task(wait_forever())

        await service.stop()

        assert service._running is False
        assert service._task is None


class TestHeartbeatServiceSendHeartbeat:
    @pytest.mark.asyncio
    async def test_records_to_event_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_store = Mock()
        service = HeartbeatService(
            event_store=mock_store,
            bot_id="test-bot",
            interval_seconds=10,
        )

        _freeze_time(monkeypatch, start=1000.0)
        await service._send_heartbeat()

        assert mock_store.store.called
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == EVENT_HEARTBEAT
        assert call_args["data"]["bot_id"] == "test-bot"
        assert call_args["data"]["count"] == 1

    @pytest.mark.asyncio
    async def test_increments_heartbeat_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=0.05)
        advance = _freeze_time(monkeypatch, start=1000.0)

        await service._send_heartbeat()
        advance(1.0)
        await service._send_heartbeat()

        assert service._heartbeat_count == 2

    @pytest.mark.asyncio
    async def test_updates_last_heartbeat_timestamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=0.05)
        _freeze_time(monkeypatch, start=1234.0)

        await service._send_heartbeat()

        assert service._last_heartbeat == 1234.0


class TestHeartbeatServiceExternalPing:
    @pytest.mark.asyncio
    async def test_ping_external_no_url(self) -> None:
        service = HeartbeatService(ping_url=None)
        result = await service._ping_external()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_external_handles_import_error(self) -> None:
        """Test that ping handles missing aiohttp gracefully."""
        service = HeartbeatService(ping_url="https://hc-ping.com/test")

        with patch.dict("sys.modules", {"aiohttp": None}):
            result = await service._ping_external()
            assert result is False


class TestHeartbeatServiceStatus:
    def test_get_status_initial(self) -> None:
        service = HeartbeatService(
            interval_seconds=30,
            ping_url="https://hc-ping.com/test",
        )

        status = service.get_status()

        assert status["enabled"] is True
        assert status["running"] is False
        assert status["interval_seconds"] == 30
        assert status["ping_url_configured"] is True
        assert status["heartbeat_count"] == 0
        assert status["last_heartbeat"] == 0.0
        assert status["seconds_since_last"] is None

    @pytest.mark.asyncio
    async def test_get_status_after_heartbeat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=5)
        advance = _freeze_time(monkeypatch, start=1000.0)
        service._running = True

        await service._send_heartbeat()
        advance(2.0)

        status = service.get_status()

        assert status["running"] is True
        assert status["heartbeat_count"] == 1
        assert status["last_heartbeat"] == 1000.0
        assert status["seconds_since_last"] == 2.0

    def test_is_healthy_when_not_running(self) -> None:
        service = HeartbeatService()
        assert service.is_healthy is False

    def test_is_healthy_just_started(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _freeze_time(monkeypatch, start=1000.0)
        service = HeartbeatService(interval_seconds=10)
        service._running = True
        service._last_heartbeat = 0.0

        assert service.is_healthy is True

    def test_is_healthy_after_heartbeat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        advance = _freeze_time(monkeypatch, start=1000.0)
        service = HeartbeatService(interval_seconds=10)
        service._running = True
        service._last_heartbeat = 1000.0

        advance(5.0)
        assert service.is_healthy is True
