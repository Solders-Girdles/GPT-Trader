"""Tests for telemetry_streaming sync stream loop."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock, patch

from gpt_trader.features.live_trade.engines.telemetry_streaming import _run_stream_loop


class TestRunStreamLoop:
    """Tests for _run_stream_loop function."""

    def test_run_stream_loop_no_broker(self) -> None:
        coordinator = Mock()
        coordinator.context.broker = None

        with patch(
            "gpt_trader.features.live_trade.engines.telemetry_streaming.logger"
        ) as mock_logger:
            _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)
            mock_logger.error.assert_called_once()

    def test_run_stream_loop_processes_messages(self) -> None:
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

    def test_run_stream_loop_routes_user_events(self) -> None:
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [{"channel": "user", "events": []}]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._user_event_handler = Mock()

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._user_event_handler.handle_user_message.assert_called_once_with(
            {"channel": "user", "events": []}
        )

    def test_run_stream_loop_triggers_backfill_on_gap(self) -> None:
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [{"channel": "ticker", "gap_detected": True}]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._user_event_handler = Mock()
        coordinator._extract_mark_from_message.return_value = None

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        reasons = [
            call.kwargs.get("reason")
            for call in coordinator._user_event_handler.request_backfill.call_args_list
        ]
        assert "startup" in reasons
        assert "sequence_gap" in reasons

    def test_run_stream_loop_stops_on_signal(self) -> None:
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
        stop_signal.set()

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, stop_signal)
        broker.stream_orderbook.assert_called_once_with(["BTC-PERP"], level=1, include_trades=True)
        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_falls_back_to_trades(self) -> None:
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

        assert coordinator._update_mark_and_metrics.call_count == 1

    def test_run_stream_loop_skips_messages_without_symbol(self) -> None:
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"bid": "50000"},
            {"product_id": "", "bid": "50000"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = 50000.5

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_skips_invalid_mark(self) -> None:
        coordinator = Mock()
        broker = Mock()
        broker.stream_orderbook.return_value = [
            {"product_id": "BTC-PERP", "bid": "50000"},
        ]
        coordinator.context.broker = broker
        coordinator.context.event_store = Mock()
        coordinator.context.bot_id = "test"
        coordinator._extract_mark_from_message.return_value = None

        _run_stream_loop(coordinator, ["BTC-PERP"], 1, None)

        coordinator._update_mark_and_metrics.assert_not_called()

    def test_run_stream_loop_skips_zero_mark(self) -> None:
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

            calls = mock_emit.call_args_list
            assert any(call[0][2].get("event_type") == "ws_stream_exit" for call in calls)
