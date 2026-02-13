"""Focused retry-state tests for telemetry streaming reconnect behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gpt_trader.features.live_trade.engines.telemetry_streaming as telemetry_streaming_module
from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    WS_STREAM_RETRY_EVENT,
    StreamingRetryState,
    _run_stream_loop,
)


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


def test_retry_state_resets_when_loop_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.stream_orderbook.side_effect = RuntimeError("always fail")
    broker.stream_trades.side_effect = RuntimeError("fallback fail")

    coordinator, _ = _create_coordinator(broker)
    coordinator._stream_retry_state = StreamingRetryState(
        base_delay=0.0,
        multiplier=1.0,
        max_delay=0.0,
        max_attempts=2,
        attempts=2,
    )

    mock_emit = Mock()
    monkeypatch.setattr(telemetry_streaming_module, "_emit_metric", mock_emit)
    monkeypatch.setattr(telemetry_streaming_module.time, "sleep", Mock())

    _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

    retry_calls = [
        call
        for call in mock_emit.call_args_list
        if call.args[2].get("event_type") == WS_STREAM_RETRY_EVENT
    ]
    assert retry_calls
    assert retry_calls[0].args[2]["attempts"] == 1


def test_retry_backoff_wait_is_interruptible(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.stream_orderbook.side_effect = RuntimeError("always fail")
    broker.stream_trades.side_effect = RuntimeError("fallback fail")

    coordinator, _ = _create_coordinator(broker)
    coordinator._stream_retry_state = StreamingRetryState(
        base_delay=5.0,
        multiplier=1.0,
        max_delay=5.0,
        max_attempts=5,
    )

    class StopAfterWait:
        def __init__(self) -> None:
            self.wait_calls: list[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, delay: float) -> bool:
            self.wait_calls.append(delay)
            return True

    stop_signal = StopAfterWait()
    mock_emit = Mock()
    sleep_mock = Mock()
    monkeypatch.setattr(telemetry_streaming_module, "_emit_metric", mock_emit)
    monkeypatch.setattr(telemetry_streaming_module.time, "sleep", sleep_mock)

    _run_stream_loop(coordinator, ["BTC-PERP"], 1, stop_signal)  # type: ignore[arg-type]

    assert stop_signal.wait_calls == [5.0]
    sleep_mock.assert_not_called()
