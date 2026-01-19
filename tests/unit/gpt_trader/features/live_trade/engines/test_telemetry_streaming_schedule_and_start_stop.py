"""Tests for telemetry_streaming coroutine scheduling and lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _schedule_coroutine,
    _start_streaming,
    _stop_streaming,
)


class TestScheduleCoroutine:
    """Tests for _schedule_coroutine function."""

    @pytest.mark.asyncio
    async def test_schedule_coroutine_with_running_loop(self) -> None:
        coordinator = Mock()
        executed = []

        async def test_coro() -> None:
            executed.append(True)

        coro = test_coro()
        _schedule_coroutine(coordinator, coro)

        await asyncio.sleep(0.01)

        assert len(executed) == 1

    def test_schedule_coroutine_no_running_loop(self) -> None:
        coordinator = Mock()
        coordinator._loop_task_handle = None
        executed = []

        async def test_coro() -> None:
            executed.append(True)

        coro = test_coro()
        _schedule_coroutine(coordinator, coro)

        assert len(executed) == 1

    def test_schedule_coroutine_via_task_handle(self) -> None:
        coordinator = Mock()

        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        mock_loop.call_soon_threadsafe = Mock()

        mock_task_handle = Mock()
        mock_task_handle.get_loop.return_value = mock_loop

        coordinator._loop_task_handle = mock_task_handle

        async def test_coro() -> None:
            pass

        coro = test_coro()
        _schedule_coroutine(coordinator, coro)
        mock_loop.call_soon_threadsafe.assert_called_once_with(asyncio.create_task, coro)
        coro.close()


class TestStartStreaming:
    """Tests for _start_streaming async function."""

    @pytest.mark.asyncio
    async def test_start_streaming_no_symbols(self) -> None:
        coordinator = Mock()
        coordinator.context.symbols = []

        result = await _start_streaming(coordinator)

        assert result is None

    @pytest.mark.asyncio
    async def test_start_streaming_creates_task(self) -> None:
        coordinator = Mock()
        coordinator.context.symbols = ["BTC-PERP", "ETH-PERP"]
        coordinator.context.config.perps_stream_level = 2
        coordinator._run_stream_loop_async = AsyncMock()
        coordinator._handle_stream_task_completion = Mock()
        coordinator._ws_stop = None

        result = await _start_streaming(coordinator)

        assert result is not None
        assert coordinator._pending_stream_config == (["BTC-PERP", "ETH-PERP"], 2)
        assert coordinator._ws_stop is not None

    @pytest.mark.asyncio
    async def test_start_streaming_default_level(self) -> None:
        coordinator = Mock()
        coordinator.context.symbols = ["BTC-PERP"]
        coordinator.context.config.perps_stream_level = None
        coordinator._run_stream_loop_async = AsyncMock()
        coordinator._handle_stream_task_completion = Mock()

        await _start_streaming(coordinator)

        assert coordinator._pending_stream_config[1] == 1

    @pytest.mark.asyncio
    async def test_start_streaming_invalid_level(self) -> None:
        coordinator = Mock()
        coordinator.context.symbols = ["BTC-PERP"]
        coordinator.context.config.perps_stream_level = "invalid"
        coordinator._run_stream_loop_async = AsyncMock()
        coordinator._handle_stream_task_completion = Mock()

        await _start_streaming(coordinator)

        assert coordinator._pending_stream_config[1] == 1


class TestStopStreaming:
    """Tests for _stop_streaming async function."""

    @pytest.mark.asyncio
    async def test_stop_streaming_clears_config(self) -> None:
        coordinator = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)
        coordinator._ws_stop = None
        coordinator._stream_task = None

        await _stop_streaming(coordinator)

        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_stop_streaming_sets_stop_signal(self) -> None:
        coordinator = Mock()
        stop_signal = threading.Event()
        coordinator._ws_stop = stop_signal
        coordinator._stream_task = None
        coordinator._pending_stream_config = None

        await _stop_streaming(coordinator)

        assert stop_signal.is_set()
        assert coordinator._ws_stop is None

    @pytest.mark.asyncio
    async def test_stop_streaming_cancels_task(self) -> None:
        coordinator = Mock()
        coordinator._ws_stop = None
        coordinator._pending_stream_config = None

        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        coordinator._stream_task = mock_task

        await _stop_streaming(coordinator)

        mock_task.cancel.assert_called_once()
        assert coordinator._stream_task is None

    @pytest.mark.asyncio
    async def test_stop_streaming_handles_cancelled_error(self) -> None:
        coordinator = Mock()
        coordinator._ws_stop = None
        coordinator._pending_stream_config = None

        async def cancelled_coro() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(cancelled_coro())
        task.cancel()

        coordinator._stream_task = task

        await _stop_streaming(coordinator)
        assert coordinator._stream_task is None
        assert coordinator._loop_task_handle is None
