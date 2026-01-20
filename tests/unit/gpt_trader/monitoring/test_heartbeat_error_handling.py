"""Error handling tests for the heartbeat service."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import gpt_trader.monitoring.heartbeat as heartbeat
from gpt_trader.monitoring.heartbeat import HeartbeatService


class TestHeartbeatServiceErrorHandling:
    """Tests for HeartbeatService error handling."""

    @pytest.mark.asyncio
    async def test_continues_on_event_store_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(interval_seconds=1)
        calls = {"count": 0}

        async def flaky_send() -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise Exception("Store error")

        async def fake_sleep(_seconds: float) -> None:
            if calls["count"] >= 2:
                service._running = False

        monkeypatch.setattr(service, "_send_heartbeat", flaky_send)
        monkeypatch.setattr(heartbeat.asyncio, "sleep", fake_sleep)

        service._running = True
        await service._heartbeat_loop()

        assert calls["count"] == 2

    @pytest.mark.asyncio
    async def test_continues_on_ping_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = HeartbeatService(
            ping_url="https://hc-ping.com/test",
            interval_seconds=1,
        )

        ping_mock = AsyncMock(return_value=False)
        monkeypatch.setattr(service, "_ping_external", ping_mock)

        await service._send_heartbeat()
        await service._send_heartbeat()

        assert service._heartbeat_count == 2
        assert ping_mock.call_count == 2
