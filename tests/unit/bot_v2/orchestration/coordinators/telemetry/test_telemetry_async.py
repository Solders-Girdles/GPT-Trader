"""Tests for async coroutine scheduling edge cases and fallback behaviors.

This module tests:
- Coroutine scheduling with running event loop
- Fallback to asyncio.run when no loop available
- Thread-safe scheduling via loop task handle
- Error handling in scheduling paths
- Compatibility methods for background operations
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator


class TestAsyncCoroutineScheduling:
    """Test async coroutine scheduling edge cases and fallback behaviors."""

    def test_schedule_coroutine_with_running_loop(self, make_context) -> None:
        """Test scheduling coroutine when event loop is running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to be running
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_loop.create_task = Mock()

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use create_task when loop is running
            mock_loop.create_task.assert_called_once()

    def test_schedule_coroutine_with_no_running_loop(self, make_context) -> None:
        """Test scheduling coroutine when no event loop is running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to track if it's called
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should call asyncio.run when no loop is available
            assert len(run_called) == 1

    def test_schedule_coroutine_with_existing_loop_task_handle(self, make_context) -> None:
        """Test scheduling coroutine with existing loop task handle."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create mock loop task handle
        mock_task_loop = Mock()
        mock_task_loop.is_running.return_value = True
        mock_task_loop.call_soon_threadsafe = Mock()

        mock_task_handle = Mock()
        mock_task_handle.get_loop.return_value = mock_task_loop

        coordinator._loop_task_handle = mock_task_handle

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use call_soon_threadsafe when task handle has running loop
            mock_task_loop.call_soon_threadsafe.assert_called_once()

    def test_schedule_coroutine_with_failing_task_handle_loop(self, make_context) -> None:
        """Test scheduling coroutine when task handle loop access fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create mock loop task handle that raises exception
        mock_task_handle = Mock()
        mock_task_handle.get_loop.side_effect = Exception("Loop access failed")

        coordinator._loop_task_handle = mock_task_handle

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to track if it's called
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should fallback to asyncio.run when task handle fails
            assert len(run_called) == 1

    def test_schedule_coroutine_with_available_loop(self, make_context) -> None:
        """Test scheduling coroutine when loop is available but not running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to be available but not running
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete = Mock()

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use run_until_complete when loop is available but not running
            mock_loop.run_until_complete.assert_called_once()

    def test_schedule_coroutine_error_handling(self, make_context) -> None:
        """Test error handling in coroutine scheduling."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError (which is the expected exception)
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to handle the fallback
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            # Should handle the error gracefully and use fallback
            coordinator._schedule_coroutine(test_coro())

            # Should call asyncio.run as fallback
            assert len(run_called) == 1

    def test_start_streaming_background_method(self, make_context) -> None:
        """Test start_streaming_background compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _schedule_coroutine to track calls
        coordinator._schedule_coroutine = Mock()

        coordinator.start_streaming_background()

        # Should schedule _start_streaming coroutine
        coordinator._schedule_coroutine.assert_called_once()
        coro_arg = coordinator._schedule_coroutine.call_args[0][0]
        # Verify it's the right coroutine (this is basic check)
        assert hasattr(coro_arg, "__await__")

    def test_stop_streaming_background_method(self, make_context) -> None:
        """Test stop_streaming_background compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _schedule_coroutine to track calls
        coordinator._schedule_coroutine = Mock()

        coordinator.stop_streaming_background()

        # Should schedule _stop_streaming coroutine
        coordinator._schedule_coroutine.assert_called_once()
        coro_arg = coordinator._schedule_coroutine.call_args[0][0]
        # Verify it's the right coroutine (this is basic check)
        assert hasattr(coro_arg, "__await__")

    @pytest.mark.asyncio
    async def test_run_account_telemetry_compatibility_method(self, make_context) -> None:
        """Test run_account_telemetry compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _run_account_telemetry to track calls
        coordinator._run_account_telemetry = AsyncMock()

        await coordinator.run_account_telemetry(interval_seconds=150)

        # Should call _run_account_telemetry with the interval
        coordinator._run_account_telemetry.assert_called_once_with(150)
