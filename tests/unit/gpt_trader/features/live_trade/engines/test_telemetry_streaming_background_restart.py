"""Tests for telemetry_streaming background start/stop and restart helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    restart_streaming_if_needed,
    start_streaming_background,
    stop_streaming_background,
)


class TestStreamingBackground:
    """Tests for start/stop streaming background functions."""

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
    """Tests for restart_streaming_if_needed function."""

    def test_no_restart_for_irrelevant_diff(self) -> None:
        coordinator = Mock()

        restart_streaming_if_needed(coordinator, {"some_other_key": "value"})

        coordinator._schedule_coroutine.assert_not_called()

    def test_restart_for_streaming_enable_change(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = lambda coro: asyncio.run(coro)

        restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "true"})

        coordinator._schedule_coroutine.assert_called()

    def test_restart_for_symbols_change(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = lambda coro: asyncio.run(coro)
        coordinator.context.symbols = ["BTC-PERP"]

        restart_streaming_if_needed(coordinator, {"symbols": ["BTC-PERP"]})

        coordinator._schedule_coroutine.assert_called()

    def test_restart_for_stream_level_change(self) -> None:
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = lambda coro: asyncio.run(coro)

        restart_streaming_if_needed(coordinator, {"perps_stream_level": 2})

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
