"""Tests for telemetry_streaming helper functions and task completion."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.utilities.telemetry as telemetry_module
from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _emit_metric,
    _handle_stream_task_completion,
    _run_stream_loop_async,
    _should_enable_streaming,
)


class TestEmitMetric:
    """Tests for _emit_metric function."""

    def test_emit_metric_calls_utility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _emit_metric calls the utility emit_metric."""
        mock_emit = MagicMock()
        monkeypatch.setattr(telemetry_module, "emit_metric", mock_emit)

        event_store = Mock()
        bot_id = "test_bot"
        payload = {"event_type": "test"}

        _emit_metric(event_store, bot_id, payload)

        mock_emit.assert_called_once_with(event_store, bot_id, payload)


class TestShouldEnableStreaming:
    """Tests for _should_enable_streaming function."""

    def test_returns_false_for_test_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "test"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_false_when_streaming_disabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = False

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_true_for_prod_with_streaming_enabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_true_for_canary_with_streaming_enabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "canary"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_false_for_dev_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "dev"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_handles_profile_with_value_attribute(self) -> None:
        coordinator = Mock()
        profile_enum = Mock()
        profile_enum.value = "prod"
        coordinator.context.config.profile = profile_enum
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_handles_none_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = None
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False


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
