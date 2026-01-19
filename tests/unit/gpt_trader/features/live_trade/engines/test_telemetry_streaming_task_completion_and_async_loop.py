"""Tests for telemetry_streaming task completion and async wrapper."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _handle_stream_task_completion,
    _run_stream_loop_async,
)


class TestHandleStreamTaskCompletion:
    """Tests for _handle_stream_task_completion function."""

    def test_clears_coordinator_state(self) -> None:
        coordinator = Mock()
        coordinator._stream_task = Mock()
        coordinator._ws_stop = Mock()

        mock_task = Mock()
        mock_task.result.return_value = None

        _handle_stream_task_completion(coordinator, mock_task)

        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handles_cancelled_error(self) -> None:
        coordinator = Mock()

        mock_task = Mock()
        mock_task.result.side_effect = asyncio.CancelledError()

        _handle_stream_task_completion(coordinator, mock_task)

        assert coordinator._stream_task is None


class TestRunStreamLoopAsync:
    """Tests for _run_stream_loop_async function."""

    @pytest.mark.asyncio
    async def test_run_stream_loop_async_calls_sync_version(self) -> None:
        coordinator = Mock()
        coordinator._run_stream_loop = Mock()

        symbols = ["BTC-PERP"]
        level = 1
        stop_signal = threading.Event()
        stop_signal.set()

        await _run_stream_loop_async(coordinator, symbols, level, stop_signal)

        coordinator._run_stream_loop.assert_called_once_with(symbols, level, stop_signal)

    @pytest.mark.asyncio
    async def test_run_stream_loop_async_handles_cancellation(self) -> None:
        coordinator = Mock()

        def slow_sync_executor(_symbols: Any, _level: Any, stop_signal: Any) -> None:
            if isinstance(stop_signal, threading.Event):
                stop_signal.wait(timeout=1.0)

        coordinator._run_stream_loop = slow_sync_executor

        symbols = ["BTC-PERP"]
        level = 1
        stop_signal = threading.Event()

        task = asyncio.create_task(_run_stream_loop_async(coordinator, symbols, level, stop_signal))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert stop_signal.is_set()
