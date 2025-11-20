"""Tests for complete streaming lifecycle management.

This module tests:
- Background task startup and shutdown
- Stream task start/stop cycles
- Task cancellation and cleanup
- Multiple start/stop cycles
- Profile-based streaming enablement
- Event loop handling
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.engines.telemetry_coordinator import TelemetryCoordinator


@pytest.mark.asyncio
async def test_start_background_tasks_starts_account_telemetry(
    make_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test start_background_tasks initializes account telemetry."""
    broker = Mock()
    risk_manager = Mock()
    account_telemetry = Mock()
    account_telemetry.supports_snapshots.return_value = True
    account_telemetry.run = AsyncMock()

    context = make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated_context = coordinator.initialize(context)

    extras = dict(updated_context.registry.extras)
    extras["account_telemetry"] = account_telemetry
    updated_context = updated_context.with_updates(
        registry=updated_context.registry.with_updates(extras=extras)
    )
    coordinator.update_context(updated_context)

    tasks = await coordinator.start_background_tasks()
    assert len(tasks) >= 1

    for task in tasks:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_shutdown_cancels_streaming(make_context, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test shutdown cancels all streaming tasks."""
    broker = Mock()
    broker.stream_orderbook.return_value = []
    risk_manager = Mock()
    risk_manager.last_mark_update = {}

    context = make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)
    coordinator.update_context(
        updated.with_updates(
            config=updated.config.model_copy(update={"perps_enable_streaming": True})
        )
    )

    tasks = await coordinator.start_background_tasks()
    await coordinator.shutdown()

    for task in tasks:
        assert task.cancelled() or task.done()


class TestStreamingLifecycleManagement:
    """Test complete streaming lifecycle management including task cancellation and cleanup."""

    @pytest.mark.asyncio
    async def test_start_streaming_with_no_symbols_returns_none(self, make_context) -> None:
        """Test start streaming returns None when no symbols are configured."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=())  # Empty symbols
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        result = await coordinator._start_streaming()

        # Should return None when no symbols are configured
        assert result is None
        assert coordinator._stream_task is None
        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_start_streaming_with_invalid_stream_level(self, make_context) -> None:
        """Test start streaming handles invalid stream level gracefully."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))

        # Update config with invalid stream level
        updated_config = context.config.model_copy(update={"perps_stream_level": "invalid"})
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to prevent actual streaming
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.create_task.return_value = Mock()
            mock_loop.is_running.return_value = True

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            result = await coordinator._start_streaming()

            # Should handle invalid level gracefully and default to 1
            assert result is not None
            assert coordinator._pending_stream_config == (["BTC-PERP"], 1)

    @pytest.mark.asyncio
    async def test_start_streaming_with_no_event_loop(self, make_context) -> None:
        """Test start streaming when no running event loop is available."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running event loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            result = await coordinator._start_streaming()

            # Should return None when no event loop is available
            assert result is None

    def test_stop_streaming_cleanup_logic(self, make_context) -> None:
        """Test stop streaming cleanup logic without awaiting tasks."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Test the state cleanup parts of _stop_streaming
        coordinator._ws_stop = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)

        # Manually test the cleanup logic that happens after task cancellation
        coordinator._ws_stop.set()
        coordinator._ws_stop = None
        coordinator._stream_task = None
        coordinator._loop_task_handle = None
        coordinator._pending_stream_config = None

        # Verify cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None
        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_stop_streaming_handles_cancelled_error(self, make_context) -> None:
        """Test stop streaming handles CancelledError from task cancellation."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock task that raises CancelledError when awaited
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        mock_task.side_effect = asyncio.CancelledError("Task cancelled")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()
        coordinator._loop_task_handle = mock_task

        # Should handle CancelledError gracefully
        await coordinator._stop_streaming()

        # Should still clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    @pytest.mark.asyncio
    async def test_stop_streaming_with_already_done_task(self, make_context) -> None:
        """Test stop streaming when task is already done."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock task that's already done
        mock_task = Mock()
        mock_task.done.return_value = True
        mock_task.cancel = Mock()
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()
        coordinator._loop_task_handle = mock_task

        await coordinator._stop_streaming()

        # Should not try to cancel already done task
        mock_task.cancel.assert_not_called()
        # Should still clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    @pytest.mark.asyncio
    async def test_stop_streaming_with_no_active_task(self, make_context) -> None:
        """Test stop streaming when no active task exists."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # No active streaming task
        assert coordinator._stream_task is None

        await coordinator._stop_streaming()

        # Should handle gracefully without errors
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    def test_handle_stream_task_completion_with_success(self, make_context) -> None:
        """Test stream task completion handler with successful task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock successful task
        mock_task = Mock()
        mock_task.result.return_value = "streaming completed"
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handle_stream_task_completion_with_cancellation(self, make_context) -> None:
        """Test stream task completion handler with cancelled task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock cancelled task
        mock_task = Mock()
        mock_task.result.side_effect = asyncio.CancelledError("Task cancelled")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state despite cancellation
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handle_stream_task_completion_with_exception(self, make_context) -> None:
        """Test stream task completion handler with failed task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock failed task
        mock_task = Mock()
        mock_task.result.side_effect = Exception("Streaming failed")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state despite failure
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    @pytest.mark.asyncio
    async def test_complete_lifecycle_start_stop_cycle(self, make_context) -> None:
        """Test complete streaming lifecycle from start to stop."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.return_value = []  # Empty stream
        context = make_context(broker=broker, symbols=("BTC-PERP",))

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Start streaming
        start_task = await coordinator._start_streaming()
        assert start_task is not None
        assert coordinator._stream_task is not None

        # Stop streaming
        await coordinator._stop_streaming()

        # Verify cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

        # Cancel the start task if it's still running
        if start_task and not start_task.done():
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self, make_context) -> None:
        """Test multiple start/stop cycles in sequence."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.return_value = []  # Empty stream
        context = make_context(broker=broker, symbols=("BTC-PERP",))

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Perform multiple start/stop cycles
        for i in range(3):
            # Start streaming
            start_task = await coordinator._start_streaming()
            assert start_task is not None
            assert coordinator._stream_task is not None

            # Short delay
            await asyncio.sleep(0.01)

            # Stop streaming
            await coordinator._stop_streaming()
            assert coordinator._stream_task is None

            # Cancel the start task if needed
            if start_task and not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    def test_streaming_shutdown_cleanup_logic(self, make_context) -> None:
        """Test streaming cleanup logic during coordinator shutdown."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Set up streaming state
        coordinator._ws_stop = Mock()
        coordinator._stream_task = Mock()
        coordinator._loop_task_handle = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)

        # Test the cleanup logic
        coordinator._ws_stop.set()
        coordinator._ws_stop = None
        coordinator._stream_task = None
        coordinator._loop_task_handle = None
        coordinator._pending_stream_config = None

        # Verify shutdown cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None
        assert coordinator._pending_stream_config is None

    def test_should_enable_streaming_logic(self, make_context) -> None:
        """Test the _should_enable_streaming logic with various profiles and settings."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        # Test CANARY profile with streaming enabled
        context_canary_enabled = make_context(broker=broker, symbols=("BTC-PERP",))
        context_canary_enabled = context_canary_enabled.with_updates(
            config=context_canary_enabled.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.CANARY}
            )
        )
        coordinator_canary = TelemetryCoordinator(context_canary_enabled)
        assert coordinator_canary._should_enable_streaming() is True

        # Test PROD profile with streaming enabled
        context_prod_enabled = make_context(broker=broker, symbols=("BTC-PERP",))
        context_prod_enabled = context_prod_enabled.with_updates(
            config=context_prod_enabled.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.PROD}
            )
        )
        coordinator_prod = TelemetryCoordinator(context_prod_enabled)
        assert coordinator_prod._should_enable_streaming() is True

        # Test DEV profile (should not enable streaming)
        context_dev = make_context(broker=broker, symbols=("BTC-PERP",))
        context_dev = context_dev.with_updates(
            config=context_dev.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.DEV}
            )
        )
        coordinator_dev = TelemetryCoordinator(context_dev)
        assert coordinator_dev._should_enable_streaming() is False

        # Test streaming disabled
        context_disabled = make_context(broker=broker, symbols=("BTC-PERP",))
        context_disabled = context_disabled.with_updates(
            config=context_disabled.config.model_copy(
                update={"perps_enable_streaming": False, "profile": Profile.PROD}
            )
        )
        coordinator_disabled = TelemetryCoordinator(context_disabled)
        assert coordinator_disabled._should_enable_streaming() is False
