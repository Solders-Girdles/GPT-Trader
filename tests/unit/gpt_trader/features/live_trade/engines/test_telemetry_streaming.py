"""Tests for telemetry_streaming - WebSocket streaming for live trading."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _emit_metric,
    _handle_stream_task_completion,
    _run_stream_loop,
    _run_stream_loop_async,
    _schedule_coroutine,
    _should_enable_streaming,
    _start_streaming,
    _stop_streaming,
    restart_streaming_if_needed,
    start_streaming_background,
    stop_streaming_background,
)

# ============================================================
# Test: _emit_metric
# ============================================================


class TestEmitMetric:
    """Tests for _emit_metric function."""

    @patch("gpt_trader.utilities.telemetry.emit_metric")
    def test_emit_metric_calls_utility(self, mock_emit: Mock) -> None:
        """Test _emit_metric calls the utility emit_metric."""
        event_store = Mock()
        bot_id = "test_bot"
        payload = {"event_type": "test"}

        _emit_metric(event_store, bot_id, payload)

        mock_emit.assert_called_once_with(event_store, bot_id, payload)


# ============================================================
# Test: _should_enable_streaming
# ============================================================


class TestShouldEnableStreaming:
    """Tests for _should_enable_streaming function."""

    def test_returns_false_for_test_profile(self) -> None:
        """Test returns False when profile is 'test'."""
        coordinator = Mock()
        coordinator.context.config.profile = "test"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_false_when_streaming_disabled(self) -> None:
        """Test returns False when perps_enable_streaming is False."""
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = False

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_true_for_prod_with_streaming_enabled(self) -> None:
        """Test returns True for prod profile with streaming enabled."""
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_true_for_canary_with_streaming_enabled(self) -> None:
        """Test returns True for canary profile with streaming enabled."""
        coordinator = Mock()
        coordinator.context.config.profile = "canary"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_false_for_dev_profile(self) -> None:
        """Test returns False for dev profile even with streaming enabled."""
        coordinator = Mock()
        coordinator.context.config.profile = "dev"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_handles_profile_with_value_attribute(self) -> None:
        """Test handles profile enum with .value attribute."""
        coordinator = Mock()
        profile_enum = Mock()
        profile_enum.value = "prod"
        coordinator.context.config.profile = profile_enum
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_handles_none_profile(self) -> None:
        """Test handles None profile."""
        coordinator = Mock()
        coordinator.context.config.profile = None
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        # None profile doesn't match prod/canary
        assert result is False


# ============================================================
# Test: start_streaming_background / stop_streaming_background
# ============================================================


class TestStreamingBackground:
    """Tests for start/stop streaming background functions."""

    def test_start_streaming_background_when_disabled(self) -> None:
        """Test start_streaming_background returns early when disabled."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False

        start_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_not_called()

    def test_start_streaming_background_when_enabled(self) -> None:
        """Test start_streaming_background schedules coroutine when enabled."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._start_streaming.return_value = AsyncMock()

        start_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_called_once()

    def test_stop_streaming_background_schedules_stop(self) -> None:
        """Test stop_streaming_background schedules stop coroutine."""
        coordinator = Mock()
        coordinator._stop_streaming.return_value = AsyncMock()

        stop_streaming_background(coordinator)

        coordinator._schedule_coroutine.assert_called_once()


# ============================================================
# Test: restart_streaming_if_needed
# ============================================================


class TestRestartStreamingIfNeeded:
    """Tests for restart_streaming_if_needed function."""

    def test_no_restart_for_irrelevant_diff(self) -> None:
        """Test no restart when diff has irrelevant keys."""
        coordinator = Mock()

        restart_streaming_if_needed(coordinator, {"some_other_key": "value"})

        coordinator._schedule_coroutine.assert_not_called()

    def test_restart_for_streaming_enable_change(self) -> None:
        """Test restart when perps_enable_streaming changes."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None

        restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "true"})

        coordinator._schedule_coroutine.assert_called()

    def test_restart_for_symbols_change(self) -> None:
        """Test restart when symbols change."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None
        coordinator.context.symbols = ["BTC-PERP"]

        restart_streaming_if_needed(coordinator, {"symbols": ["BTC-PERP"]})

        coordinator._schedule_coroutine.assert_called()

    def test_restart_for_stream_level_change(self) -> None:
        """Test restart when stream level changes."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = True
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming.return_value = None

        restart_streaming_if_needed(coordinator, {"perps_stream_level": 2})

        coordinator._schedule_coroutine.assert_called()

    def test_handles_runtime_error_with_asyncio_run(self) -> None:
        """Test handles RuntimeError about asyncio.run in running loop."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._start_streaming = Mock()
        coordinator._schedule_coroutine.side_effect = RuntimeError(
            "asyncio.run() cannot be called from a running event loop"
        )

        # Should not raise - falls back to thread
        restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "false"})
        coordinator._schedule_coroutine.assert_called_once()
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_not_called()

    def test_reraises_other_runtime_errors(self) -> None:
        """Test reraises RuntimeError that's not about asyncio.run."""
        coordinator = Mock()
        coordinator._should_enable_streaming.return_value = False
        coordinator._stop_streaming.return_value = None
        coordinator._schedule_coroutine.side_effect = RuntimeError("Some other error")

        with pytest.raises(RuntimeError, match="Some other error"):
            restart_streaming_if_needed(coordinator, {"perps_enable_streaming": "false"})


# ============================================================
# Test: _schedule_coroutine
# ============================================================


class TestScheduleCoroutine:
    """Tests for _schedule_coroutine function."""

    @pytest.mark.asyncio
    async def test_schedule_coroutine_with_running_loop(self) -> None:
        """Test schedules coroutine when event loop is running."""
        coordinator = Mock()
        executed = []

        async def test_coro() -> None:
            executed.append(True)

        coro = test_coro()
        _schedule_coroutine(coordinator, coro)

        # Give task time to execute
        await asyncio.sleep(0.01)

        assert len(executed) == 1

    def test_schedule_coroutine_no_running_loop(self) -> None:
        """Test handles case when no running loop."""
        coordinator = Mock()
        coordinator._loop_task_handle = None
        executed = []

        async def test_coro() -> None:
            executed.append(True)

        coro = test_coro()
        _schedule_coroutine(coordinator, coro)

        # asyncio.run should have executed it
        assert len(executed) == 1

    def test_schedule_coroutine_via_task_handle(self) -> None:
        """Test schedules via loop_task_handle when available."""
        coordinator = Mock()

        # Create a mock task handle with a running loop
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        mock_loop.call_soon_threadsafe = Mock()

        mock_task_handle = Mock()
        mock_task_handle.get_loop.return_value = mock_loop

        coordinator._loop_task_handle = mock_task_handle

        async def test_coro() -> None:
            pass

        # We can't easily test this in sync context, just verify no crash
        # The function will try running loop first, and since we're not in
        # an async context, it will fall through to the task_handle path
        coro = test_coro()
        _schedule_coroutine(coordinator, coro)
        mock_loop.call_soon_threadsafe.assert_called_once_with(asyncio.create_task, coro)


# ============================================================
# Test: _start_streaming
# ============================================================


class TestStartStreaming:
    """Tests for _start_streaming async function."""

    @pytest.mark.asyncio
    async def test_start_streaming_no_symbols(self) -> None:
        """Test returns None when no symbols configured."""
        coordinator = Mock()
        coordinator.context.symbols = []

        result = await _start_streaming(coordinator)

        assert result is None

    @pytest.mark.asyncio
    async def test_start_streaming_creates_task(self) -> None:
        """Test creates streaming task with symbols."""
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
        """Test defaults to level 1 when not configured."""
        coordinator = Mock()
        coordinator.context.symbols = ["BTC-PERP"]
        coordinator.context.config.perps_stream_level = None
        coordinator._run_stream_loop_async = AsyncMock()
        coordinator._handle_stream_task_completion = Mock()

        await _start_streaming(coordinator)

        assert coordinator._pending_stream_config[1] == 1

    @pytest.mark.asyncio
    async def test_start_streaming_invalid_level(self) -> None:
        """Test handles invalid stream level gracefully."""
        coordinator = Mock()
        coordinator.context.symbols = ["BTC-PERP"]
        coordinator.context.config.perps_stream_level = "invalid"
        coordinator._run_stream_loop_async = AsyncMock()
        coordinator._handle_stream_task_completion = Mock()

        await _start_streaming(coordinator)

        # Should default to 1
        assert coordinator._pending_stream_config[1] == 1


# ============================================================
# Test: _stop_streaming
# ============================================================


class TestStopStreaming:
    """Tests for _stop_streaming async function."""

    @pytest.mark.asyncio
    async def test_stop_streaming_clears_config(self) -> None:
        """Test stop_streaming clears pending config."""
        coordinator = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)
        coordinator._ws_stop = None
        coordinator._stream_task = None

        await _stop_streaming(coordinator)

        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_stop_streaming_sets_stop_signal(self) -> None:
        """Test stop_streaming sets the stop signal."""
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
        """Test stop_streaming cancels running task."""
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
        """Test stop_streaming handles CancelledError gracefully."""
        coordinator = Mock()
        coordinator._ws_stop = None
        coordinator._pending_stream_config = None

        # Create a real cancelled task
        async def cancelled_coro() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(cancelled_coro())
        task.cancel()

        coordinator._stream_task = task

        # Should not raise
        await _stop_streaming(coordinator)
        assert coordinator._stream_task is None
        assert coordinator._loop_task_handle is None


# ============================================================
# Test: _handle_stream_task_completion
# ============================================================


class TestHandleStreamTaskCompletion:
    """Tests for _handle_stream_task_completion function."""

    def test_clears_coordinator_state(self) -> None:
        """Test clears coordinator state on completion."""
        coordinator = Mock()
        coordinator._stream_task = Mock()
        coordinator._ws_stop = Mock()

        mock_task = Mock()
        mock_task.result.return_value = None

        _handle_stream_task_completion(coordinator, mock_task)

        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handles_cancelled_error(self) -> None:
        """Test handles CancelledError from task."""
        coordinator = Mock()

        mock_task = Mock()
        mock_task.result.side_effect = asyncio.CancelledError()

        # Should not raise
        _handle_stream_task_completion(coordinator, mock_task)

        assert coordinator._stream_task is None


# ============================================================
# Test: _run_stream_loop_async
# ============================================================


class TestRunStreamLoopAsync:
    """Tests for _run_stream_loop_async function."""

    @pytest.mark.asyncio
    async def test_run_stream_loop_async_calls_sync_version(self) -> None:
        """Test async version calls sync version in executor."""
        coordinator = Mock()
        coordinator._run_stream_loop = Mock()

        symbols = ["BTC-PERP"]
        level = 1
        stop_signal = threading.Event()

        # Run briefly
        stop_signal.set()

        await _run_stream_loop_async(coordinator, symbols, level, stop_signal)

        coordinator._run_stream_loop.assert_called_once_with(symbols, level, stop_signal)

    @pytest.mark.asyncio
    async def test_run_stream_loop_async_handles_cancellation(self) -> None:
        """Test sets stop signal on cancellation."""
        coordinator = Mock()

        import time

        def slow_sync_executor(*args: Any) -> None:
            time.sleep(0.5)  # Short sleep for test efficiency

        coordinator._run_stream_loop = slow_sync_executor

        symbols = ["BTC-PERP"]
        level = 1
        stop_signal = threading.Event()

        task = asyncio.create_task(_run_stream_loop_async(coordinator, symbols, level, stop_signal))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Verify stop signal was set
        assert stop_signal.is_set()


# ============================================================
# Test: _run_stream_loop
# ============================================================


class TestRunStreamLoop:
    """Tests for _run_stream_loop function."""

    def test_run_stream_loop_no_broker(self) -> None:
        """Test returns early when no broker available."""
        coordinator = Mock()
        coordinator.context.broker = None

        with patch(
            "gpt_trader.features.live_trade.engines.telemetry_streaming.logger"
        ) as mock_logger:
            _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)
            mock_logger.error.assert_called_once()

    def test_run_stream_loop_processes_messages(self) -> None:
        """Test processes stream messages."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"product_id": "BTC-PERP", "bid": "50000", "ask": "50001"},
            {"product_id": "ETH-PERP", "bid": "3000", "ask": "3001"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 50000.5
        coordinator._update_mark_and_metrics = Mock()

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        assert coordinator._update_mark_and_metrics.call_count >= 1

    def test_run_stream_loop_stops_on_signal(self) -> None:
        """Test stops when stop signal is set."""
        coordinator = Mock()
        broker = Mock()

        def infinite_stream() -> Any:
            while True:
                yield {"product_id": "BTC-PERP", "bid": "50000"}

        broker.stream_orderbook.return_value = infinite_stream()
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 50000.5
        coordinator._update_mark_and_metrics = Mock()

        stop_signal = threading.Event()
        stop_signal.set()  # Set immediately

        # Should stop quickly
        _run_stream_loop(coordinator, ["BTC-PERP"], 1, stop_signal)
        broker.stream_orderbook.assert_called_once_with(["BTC-PERP"], level=1, include_trades=True)
        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_falls_back_to_trades(self) -> None:
        """Test falls back to stream_trades when orderbook fails."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.side_effect = Exception("Orderbook not available")
        broker.stream_trades.return_value = []
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        broker.stream_trades.assert_called_once_with(["BTC-PERP"])

    def test_run_stream_loop_skips_non_dict_messages(self) -> None:
        """Test skips non-dict messages."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            "not a dict",
            None,
            123,
            {"product_id": "BTC-PERP", "bid": "50000"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 50000.5
        coordinator._update_mark_and_metrics = Mock()

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        # Should only process the valid dict message
        assert coordinator._update_mark_and_metrics.call_count == 1

    def test_run_stream_loop_skips_messages_without_symbol(self) -> None:
        """Test skips messages without product_id or symbol."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"bid": "50000"},  # No symbol
            {"product_id": "", "bid": "50000"},  # Empty symbol
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 50000.5

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_skips_invalid_mark(self) -> None:
        """Test skips messages with invalid mark price."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"product_id": "BTC-PERP", "bid": "50000"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = None  # Invalid mark

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_skips_zero_mark(self) -> None:
        """Test skips messages with zero mark price."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"product_id": "BTC-PERP", "bid": "50000"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 0

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_emits_exit_metric(self) -> None:
        """Test emits exit metric when loop completes."""
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = []
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"

        with patch(
            "gpt_trader.features.live_trade.engines.telemetry_streaming._emit_metric"
        ) as mock_emit:
            _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

            # Should emit exit metric
            calls = mock_emit.call_args_list
            assert any(call[0][2].get("event_type") == "ws_stream_exit" for call in calls)
