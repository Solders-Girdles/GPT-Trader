"""Chaos tests for TradingEngine WebSocket health degradation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from tests.support.chaos import (
    ChaosBroker,
    ws_disconnect_scenario,
    ws_gap_scenario,
    ws_reconnect_scenario,
    ws_stale_heartbeat_scenario,
    ws_stale_messages_scenario,
)

pytest_plugins = ["strategy_engine_chaos_fixtures"]


class TestWSHealthDegradation:
    """Tests for WebSocket health-triggered degradation."""

    @pytest.mark.asyncio
    async def test_ws_stale_messages_triggers_pause_and_reduce_only(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale WS messages should trigger pause + reduce-only mode."""
        # Set up chaos broker that returns stale message health
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_stale_messages_scenario(stale_age_seconds=30.0),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered
        assert engine._degradation.is_paused()
        assert "ws_message_stale" in (engine._degradation.get_pause_reason() or "")
        engine.context.risk_manager.set_reduce_only_mode.assert_called_with(
            True, reason="ws_message_stale"
        )

    @pytest.mark.asyncio
    async def test_ws_stale_heartbeat_triggers_pause(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale WS heartbeat should trigger pause + reduce-only mode."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_stale_heartbeat_scenario(stale_age_seconds=60.0),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered
        assert engine._degradation.is_paused()
        assert "ws_heartbeat_stale" in (engine._degradation.get_pause_reason() or "")

    @pytest.mark.asyncio
    async def test_ws_reconnect_triggers_pause_for_sync(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WS reconnect should trigger a brief pause for state synchronization."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_reconnect_scenario(reconnect_count=1),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify pause was triggered for sync
        assert engine._degradation.is_paused()
        assert "ws_reconnect" in (engine._degradation.get_pause_reason() or "")
        # Verify reconnect tracking was reset
        assert engine._ws_reconnect_attempts == 0
        events = engine._event_store.list_events()
        assert any(e.get("type") == "websocket_reconnect" for e in events)

    @pytest.mark.asyncio
    async def test_ws_disconnect_with_stale_timestamps_triggers_degradation(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WS disconnect (with stale timestamps) should trigger degradation."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_disconnect_scenario(),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered (due to stale timestamps)
        assert engine._degradation.is_paused()

    @pytest.mark.asyncio
    async def test_ws_gap_count_tracked_in_status(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WS gaps should be tracked and reported in status."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_gap_scenario(gap_count=5),
        )
        engine.running = True
        engine._cycle_count = 60  # Trigger the gap logging condition

        # Mock the status reporter's update_ws_health method
        engine._status_reporter.update_ws_health = MagicMock()

        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify status reporter was updated with gap count
        engine._status_reporter.update_ws_health.assert_called()
        call_args = engine._status_reporter.update_ws_health.call_args[0][0]
        assert call_args.get("gap_count") == 5

    @pytest.mark.asyncio
    async def test_no_degradation_when_broker_lacks_ws_health(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No degradation when broker doesn't support get_ws_health."""
        # Remove get_ws_health method
        del mock_broker.get_ws_health
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # No degradation should have been triggered
        assert not engine._degradation.is_paused()

    @pytest.mark.asyncio
    async def test_ws_health_exception_handled_gracefully(
        self, engine, mock_broker, mock_risk_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exceptions in get_ws_health should be handled gracefully."""
        mock_broker.get_ws_health.side_effect = Exception("WS health error")
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        # Should not raise, exception should be caught
        monkeypatch.setattr(asyncio, "sleep", stop_after_one)
        with pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # No degradation from the exception itself
        assert not engine._degradation.is_paused()
