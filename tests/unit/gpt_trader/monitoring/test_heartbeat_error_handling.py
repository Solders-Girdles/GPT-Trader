"""Error handling tests for the heartbeat service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gpt_trader.monitoring.heartbeat import HeartbeatService

pytestmark = pytest.mark.legacy_modernize


class TestHeartbeatServiceErrorHandling:
    """Tests for HeartbeatService error handling."""

    @pytest.mark.asyncio
    async def test_continues_on_event_store_error(self) -> None:
        mock_store = Mock()
        mock_store.store.side_effect = Exception("Store error")

        service = HeartbeatService(
            event_store=mock_store,
            interval_seconds=0.05,
        )

        await service.start()
        await asyncio.sleep(0.12)
        await service.stop()

        # Should have attempted multiple heartbeats despite errors
        assert mock_store.store.call_count >= 2

    @pytest.mark.asyncio
    async def test_continues_on_ping_error(self) -> None:
        """Test that heartbeat continues even when ping fails."""
        service = HeartbeatService(
            ping_url="https://hc-ping.com/test",
            interval_seconds=0.05,
        )

        # Patch _ping_external to always fail
        with patch.object(service, "_ping_external", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = False

            await service.start()
            await asyncio.sleep(0.12)
            await service.stop()

            # Should have continued despite ping failures
            assert service._heartbeat_count >= 2
            assert mock_ping.call_count >= 2
