"""Tests for Coinbase WebSocket event types, SequenceGuard, and health monitoring."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket, SequenceGuard
from gpt_trader.features.brokerages.coinbase.ws_events import EventDispatcher, EventType


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_values(self) -> None:
        assert EventType.TICKER.value == "ticker"
        assert EventType.LEVEL2.value == "l2_data"
        assert EventType.USER.value == "user"
        assert EventType.ERROR.value == "error"


class TestSequenceGuard:
    """Tests for SequenceGuard gap detection."""

    def test_annotate_returns_message_unchanged_without_sequence(self) -> None:
        guard = SequenceGuard()
        message = {"type": "ticker", "price": "50000"}

        result = guard.annotate(message)

        assert result == message
        assert "gap_detected" not in result

    def test_annotate_first_message_no_gap(self) -> None:
        guard = SequenceGuard()
        message = {"sequence": 1, "type": "ticker"}

        result = guard.annotate(message)

        assert "gap_detected" not in result

    def test_annotate_sequential_messages_no_gap(self) -> None:
        guard = SequenceGuard()

        msg1 = guard.annotate({"sequence": 1})
        msg2 = guard.annotate({"sequence": 2})
        msg3 = guard.annotate({"sequence": 3})

        assert "gap_detected" not in msg1
        assert "gap_detected" not in msg2
        assert "gap_detected" not in msg3

    def test_annotate_gap_detected(self) -> None:
        guard = SequenceGuard()

        guard.annotate({"sequence": 1})
        result = guard.annotate({"sequence": 5})

        assert result.get("gap_detected") is True

    def test_reset_clears_state(self) -> None:
        guard = SequenceGuard()

        guard.annotate({"sequence": 100})
        guard.reset()

        result = guard.annotate({"sequence": 1})
        assert "gap_detected" not in result


class TestCoinbaseWebSocketHealth:
    """Tests for CoinbaseWebSocket.get_health()."""

    def test_get_health_returns_all_fields(self) -> None:
        ws = CoinbaseWebSocket()

        health = ws.get_health()

        assert "connected" in health
        assert "last_message_ts" in health
        assert "last_heartbeat_ts" in health
        assert "last_close_ts" in health
        assert "last_error_ts" in health
        assert "gap_count" in health
        assert "reconnect_count" in health

    def test_get_health_initial_state(self) -> None:
        ws = CoinbaseWebSocket()

        health = ws.get_health()

        assert health["connected"] is False
        assert health["last_message_ts"] is None
        assert health["last_heartbeat_ts"] is None
        assert health["last_close_ts"] is None
        assert health["last_error_ts"] is None
        assert health["gap_count"] == 0
        assert health["reconnect_count"] == 0

    def test_get_health_reflects_gap_count(self) -> None:
        ws = CoinbaseWebSocket()
        ws._gap_count = 5

        health = ws.get_health()

        assert health["gap_count"] == 5

    def test_get_health_reflects_reconnect_count(self) -> None:
        ws = CoinbaseWebSocket()
        ws._reconnect_count = 3

        health = ws.get_health()

        assert health["reconnect_count"] == 3


class TestEventDispatcher:
    """Tests for EventDispatcher."""

    def test_dispatch_ticker(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_ticker(lambda e: received.append(e))

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1
        assert received[0].product_id == "BTC-USD"

    def test_dispatch_trades(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_trade(lambda e: received.append(e))

        message = {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "ETH-USD",
                            "trade_id": "t1",
                            "price": "2500",
                            "size": "1",
                            "side": "buy",
                        }
                    ]
                }
            ],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1
        assert received[0].product_id == "ETH-USD"

    def test_dispatch_orderbook(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_orderbook(lambda e: received.append(e))

        message = {
            "channel": "l2_data",
            "events": [{"updates": [{"side": "bid", "price_level": "100", "new_quantity": "10"}]}],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1

    def test_dispatch_raw_always_called(self) -> None:
        dispatcher = EventDispatcher()
        raw_messages = []

        dispatcher.on_raw(lambda m: raw_messages.append(m))

        message = {"channel": "ticker", "data": "test"}
        dispatcher.dispatch(message)

        assert len(raw_messages) == 1
        assert raw_messages[0] == message

    def test_dispatch_error_event(self) -> None:
        dispatcher = EventDispatcher()
        errors = []

        dispatcher.on_error(lambda e: errors.append(e))

        message = {"type": "error", "message": "subscription failed"}
        dispatcher.dispatch(message)

        assert len(errors) == 1
        assert errors[0]["message"] == "subscription failed"

    def test_handler_exception_does_not_stop_dispatch(self) -> None:
        dispatcher = EventDispatcher()
        results = []

        def failing_handler(e):
            raise ValueError("handler error")

        def working_handler(e):
            results.append(e)

        dispatcher.on_ticker(failing_handler)
        dispatcher.on_ticker(working_handler)

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)

        assert len(results) == 1

    def test_multiple_handlers(self) -> None:
        dispatcher = EventDispatcher()
        results1 = []
        results2 = []

        dispatcher.on_ticker(lambda e: results1.append(e))
        dispatcher.on_ticker(lambda e: results2.append(e))

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)

        assert len(results1) == 1
        assert len(results2) == 1
