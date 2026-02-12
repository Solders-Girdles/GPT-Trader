"""Tests for telemetry_streaming helper functions and task completion."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.utilities.telemetry as telemetry_module
import gpt_trader.features.live_trade.engines.telemetry_streaming as telemetry_streaming_module
from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _emit_metric,
    _handle_stream_task_completion,
    _run_stream_loop,
    _run_stream_loop_async,
    _schedule_coroutine,
    _should_enable_streaming,
    StreamingRetryState,
    WS_STREAM_RETRY_EVENT,
    WS_STREAM_RETRY_EXHAUSTED_EVENT,
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


def _create_coordinator(broker: Mock | None = None) -> tuple[Mock, SimpleNamespace]:
    coordinator = Mock()
    context = SimpleNamespace(
        broker=broker or Mock(),
        event_store=Mock(),
        bot_id="telemetry-test-bot",
    )
    coordinator.context = context
    coordinator._extract_mark_from_message = Mock(return_value=50000.5)
    coordinator._update_mark_and_metrics = Mock()
    return coordinator, context


class TestRunStreamLoopRetryBackoff:
    def test_reconnect_backoff_progression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broker = Mock()

        def success_stream() -> Any:
            yield {"channel": "ticker", "product_id": "BTC-PERP", "bid": "50000", "ask": "50001"}

        broker.stream_orderbook.side_effect = [RuntimeError("boot"), success_stream()]
        broker.stream_trades.side_effect = RuntimeError("trades unavailable")

        coordinator, context = _create_coordinator(broker)
        mock_emit = Mock()
        monkeypatch.setattr(telemetry_streaming_module, "_emit_metric", mock_emit)
        sleep_mock = Mock()
        monkeypatch.setattr(telemetry_streaming_module.time, "sleep", sleep_mock)

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        sleep_mock.assert_called_once_with(0.5)
        retry_calls = [
            call
            for call in mock_emit.call_args_list
            if call.args[2].get("event_type") == WS_STREAM_RETRY_EVENT
        ]
        assert len(retry_calls) == 1
        assert retry_calls[0].args[2]["delay_seconds"] == 0.5
        assert retry_calls[0].args[2]["gap_count"] == 0
        assert coordinator._stream_retry_state.attempts == 0

    def test_gap_count_and_backfill_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broker = Mock()

        def failing_stream() -> Any:
            yield {"channel": "ticker", "product_id": "BTC-PERP", "bid": "50000", "ask": "50001", "gap_detected": True}
            raise RuntimeError("stream drop")

        def success_stream() -> Any:
            yield {"channel": "ticker", "product_id": "BTC-PERP", "bid": "50000", "ask": "50001"}

        broker.stream_orderbook.side_effect = [failing_stream(), success_stream()]
        broker.stream_trades.side_effect = RuntimeError("trades unavailable")

        coordinator, context = _create_coordinator(broker)
        user_handler = Mock()
        user_handler.request_backfill = Mock()
        coordinator._user_event_handler = user_handler

        mock_emit = Mock()
        monkeypatch.setattr(telemetry_streaming_module, "_emit_metric", mock_emit)
        sleep_mock = Mock()
        monkeypatch.setattr(telemetry_streaming_module.time, "sleep", sleep_mock)

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        retry_calls = [
            call for call in mock_emit.call_args_list if call.args[2].get("event_type") == WS_STREAM_RETRY_EVENT
        ]
        assert retry_calls
        assert retry_calls[0].args[2]["gap_count"] == 1
        reasons = [call.kwargs.get("reason") for call in user_handler.request_backfill.call_args_list]
        assert "sequence_gap" in reasons
        assert coordinator._stream_retry_state.attempts == 0

    def test_retry_exhaustion_emits_diagnostic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broker = Mock()
        broker.stream_orderbook.side_effect = RuntimeError("always fail")
        broker.stream_trades.side_effect = RuntimeError("fallback fail")

        coordinator, context = _create_coordinator(broker)
        coordinator._stream_retry_state = StreamingRetryState(
            base_delay=0.0, multiplier=1.0, max_delay=0.0, max_attempts=2
        )

        mock_emit = Mock()
        monkeypatch.setattr(telemetry_streaming_module, "_emit_metric", mock_emit)
        sleep_mock = Mock()
        monkeypatch.setattr(telemetry_streaming_module.time, "sleep", sleep_mock)

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        assert sleep_mock.call_count == 2
        exhausted_calls = [
            call
            for call in mock_emit.call_args_list
            if call.args[2].get("event_type") == WS_STREAM_RETRY_EXHAUSTED_EVENT
        ]
        assert exhausted_calls
        assert exhausted_calls[0].args[2]["attempts"] == 2
