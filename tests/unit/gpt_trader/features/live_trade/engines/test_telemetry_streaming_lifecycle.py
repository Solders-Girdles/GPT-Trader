"""Tests for telemetry_streaming coroutine scheduling and lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _start_streaming,
    _stop_streaming,
    restart_streaming_if_needed,
    start_streaming_background,
    stop_streaming_background,
)


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


class TestStreamingBackground:
    def test_start_streaming_background_when_disabled(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False

        start_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_not_called()

    def test_start_streaming_background_when_enabled(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._start_streaming.return_value = AsyncMock()

        start_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_called_once()

    def test_stop_streaming_background_schedules_stop(self) -> None:
        coordinator = Mock()
        coordinator._stop_streaming.return_value = AsyncMock()

        stop_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_called_once()


class TestRestartStreamingIfNeeded:
    def test_no_restart_for_irrelevant_diff(self) -> None:
        coordinator = Mock()

        restart_streaming_if_needed(coordinator, {"some_other_key": "value"})

        coordinator._schedule_coroutine.assert_not_called()

    @pytest.mark.parametrize(
        ("diff", "should_enable"),
        [
            ({"perps_enable_streaming": "true"}, False),
            ({"symbols": ["BTC-PERP"]}, True),
            ({"perps_stream_level": 2}, True),
        ],
    )
    def test_restart_for_config_change(self, diff: dict, should_enable: bool) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = should_enable
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = lambda coro: asyncio.run(coro)
        coordinator.context.symbols = ["BTC-PERP"]

        restart_streaming_if_needed(coordinator, diff)

        coordinator._schedule_coroutine.assert_called()

    def test_handles_runtime_error_with_asyncio_run(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming = Mock()
        coordinator._schedule_coroutine.side_effect = RuntimeError(
            "asyncio.run() cannot be called from a running event loop"
        )

        restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "false"})
        coordinator._schedule_coroutine.assert_called_once()
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_not_called()

    def test_reraises_other_runtime_errors(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = RuntimeError("Some other error")

        with pytest.raises(RuntimeError, match="Some other error"):
            restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "false"})
