"""Unit tests for the heartbeat service."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from gpt_trader.monitoring.heartbeat import EVENT_HEARTBEAT, HeartbeatService


class TestHeartbeatServiceInit:
    """Tests for HeartbeatService initialization."""

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
    """Tests for HeartbeatService start method."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self) -> None:
        service = HeartbeatService(enabled=False)
        task = await service.start()
        assert task is None
        assert service._running is False

    @pytest.mark.asyncio
    async def test_start_creates_task(self) -> None:
        service = HeartbeatService(interval_seconds=1)
        task = await service.start()

        try:
            assert task is not None
            assert service._running is True
            assert service._task is task
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self) -> None:
        service = HeartbeatService(interval_seconds=1)
        task1 = await service.start()
        task2 = await service.start()

        try:
            assert task1 is task2  # Should return same task
        finally:
            await service.stop()


class TestHeartbeatServiceStop:
    """Tests for HeartbeatService stop method."""

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        service = HeartbeatService()
        await service.stop()  # Should not raise
        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        service = HeartbeatService(interval_seconds=10)
        await service.start()

        await service.stop()

        assert service._running is False
        assert service._task is None


class TestHeartbeatServiceSendHeartbeat:
    """Tests for HeartbeatService heartbeat sending."""

    @pytest.mark.asyncio
    async def test_records_to_event_store(self) -> None:
        mock_store = Mock()
        service = HeartbeatService(
            event_store=mock_store,
            bot_id="test-bot",
            interval_seconds=0.1,
        )

        await service.start()
        await asyncio.sleep(0.15)  # Allow one heartbeat
        await service.stop()

        assert mock_store.store.called
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == EVENT_HEARTBEAT
        assert call_args["data"]["bot_id"] == "test-bot"
        assert call_args["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_increments_heartbeat_count(self) -> None:
        service = HeartbeatService(interval_seconds=0.05)

        await service.start()
        await asyncio.sleep(0.12)  # Allow ~2 heartbeats
        await service.stop()

        assert service._heartbeat_count >= 2

    @pytest.mark.asyncio
    async def test_updates_last_heartbeat_timestamp(self) -> None:
        service = HeartbeatService(interval_seconds=0.05)

        before = time.time()
        await service.start()
        await asyncio.sleep(0.1)
        await service.stop()
        after = time.time()

        assert before <= service._last_heartbeat <= after


class TestHeartbeatServiceExternalPing:
    """Tests for HeartbeatService external ping functionality."""

    @pytest.mark.asyncio
    async def test_ping_external_no_url(self) -> None:
        service = HeartbeatService(ping_url=None)
        result = await service._ping_external()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_external_handles_import_error(self) -> None:
        """Test that ping handles missing aiohttp gracefully."""
        service = HeartbeatService(ping_url="https://hc-ping.com/test")

        # Clear aiohttp from modules to simulate it not being installed
        with patch.dict("sys.modules", {"aiohttp": None}):
            # This should not raise, just return False
            result = await service._ping_external()
            # Will fail to import, so returns False
            assert result is False


class TestHeartbeatServiceStatus:
    """Tests for HeartbeatService status methods."""

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
    async def test_get_status_after_heartbeat(self) -> None:
        service = HeartbeatService(interval_seconds=0.05)

        await service.start()
        await asyncio.sleep(0.1)

        status = service.get_status()

        assert status["running"] is True
        assert status["heartbeat_count"] >= 1
        assert status["last_heartbeat"] > 0
        assert status["seconds_since_last"] is not None
        assert status["seconds_since_last"] < 1.0

        await service.stop()

    def test_is_healthy_when_not_running(self) -> None:
        service = HeartbeatService()
        assert service.is_healthy is False

    @pytest.mark.asyncio
    async def test_is_healthy_just_started(self) -> None:
        service = HeartbeatService(interval_seconds=10)
        await service.start()

        # Just started, no heartbeat yet - should be healthy
        assert service.is_healthy is True

        await service.stop()

    @pytest.mark.asyncio
    async def test_is_healthy_after_heartbeat(self) -> None:
        service = HeartbeatService(interval_seconds=0.05)
        await service.start()
        await asyncio.sleep(0.1)

        assert service.is_healthy is True

        await service.stop()
