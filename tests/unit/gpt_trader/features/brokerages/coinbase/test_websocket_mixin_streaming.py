"""Unit tests for WebSocketClientMixin streaming and message filtering."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client.websocket_mixin as websocket_mixin_module
from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import _STREAM_STOP
from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
)


def _run_stream(thread_target, subscribed: threading.Event, stop_event: threading.Event) -> None:
    thread = threading.Thread(target=thread_target)
    thread.start()
    assert subscribed.wait(timeout=1.0)
    stop_event.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


class TestStreamOrderbook:
    """Tests for stream_orderbook method."""

    @pytest.mark.parametrize(
        ("level", "include_trades", "include_user_events", "expected_channels"),
        [
            (2, False, False, ["level2"]),
            (1, False, False, ["ticker"]),
            (2, True, False, ["level2", "market_trades"]),
            (1, False, True, ["ticker"]),
        ],
    )
    def test_stream_orderbook_subscriptions(
        self,
        level: int,
        include_trades: bool,
        include_user_events: bool,
        expected_channels: list[str],
        mock_websocket_class: MagicMock,
    ) -> None:
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws
        client = MockWebSocketClient()
        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(
                ["BTC-USD"],
                level=level,
                stop_event=stop_event,
                include_trades=include_trades,
                include_user_events=include_user_events,
            ):
                pass

        _run_stream(consume_stream, subscribed, stop_event)

        mock_ws.connect.assert_called_once()
        mock_ws.subscribe.assert_called_once_with(["BTC-USD"], expected_channels)
        if include_user_events:
            mock_ws.subscribe_user_events.assert_called_once_with(["BTC-USD"])
        else:
            mock_ws.subscribe_user_events.assert_not_called()

    def test_yields_messages_from_callback(self):
        """Messages pushed via callback should be yielded by stream."""
        client = MockWebSocketClient()
        client._stream_active = True
        client._on_websocket_message({"type": "level2", "product_id": "BTC-USD", "price": "50000"})
        client._on_websocket_message({"type": "level2", "product_id": "BTC-USD", "price": "50001"})
        msg1 = client._message_queue.get(timeout=1)
        msg2 = client._message_queue.get(timeout=1)
        assert msg1["price"] == "50000"
        assert msg2["price"] == "50001"


class TestStreamTrades:
    """Tests for stream_trades method."""

    def test_connects_and_subscribes_to_market_trades(self, mock_websocket_class: MagicMock):
        """stream_trades should subscribe to market_trades channel."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()

        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_trades(["BTC-USD", "ETH-USD"], stop_event=stop_event):
                pass

        _run_stream(consume_stream, subscribed, stop_event)

        mock_ws.connect.assert_called_once()
        mock_ws.subscribe.assert_called_once_with(["BTC-USD", "ETH-USD"], ["market_trades"])

    def test_yields_trade_messages(self, mock_websocket_class: MagicMock):
        """Trade messages should be yielded via the queue mechanism."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()
        client._stream_active = True

        client._on_websocket_message(
            {"type": "market_trades", "product_id": "BTC-USD", "price": "49999", "size": "0.1"}
        )

        msg = client._message_queue.get(timeout=1)

        assert msg["price"] == "49999"
        assert msg["size"] == "0.1"


class TestMessageFiltering:
    """Tests for message filtering behavior."""

    def test_ignores_non_dict_messages(self, mock_websocket_class: MagicMock):
        """Non-dict messages should be ignored."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws
        client = MockWebSocketClient()
        received_messages = []
        stop_event = threading.Event()
        received_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for msg in client.stream_orderbook(["BTC-USD"], stop_event=stop_event):
                received_messages.append(msg)
                received_event.set()

        thread = threading.Thread(target=consume_stream)
        thread.start()
        assert subscribed.wait(timeout=1.0)
        client._message_queue.put("string message")
        client._message_queue.put(123)
        client._message_queue.put({"valid": "message"})
        assert received_event.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(received_messages) == 1
        assert received_messages[0] == {"valid": "message"}

    def test_on_websocket_message_ignored_when_inactive(self):
        """Messages should be ignored when stream is inactive."""
        client = MockWebSocketClient()
        client._stream_active = False
        client._on_websocket_message({"type": "ticker"})
        assert client._message_queue.empty()

    def test_clears_queue_before_streaming(self):
        """_clear_queue should remove all pending messages."""
        client = MockWebSocketClient()
        client._message_queue.put({"stale": "message1"})
        client._message_queue.put({"stale": "message2"})
        assert not client._message_queue.empty()
        client._clear_queue()
        assert client._message_queue.empty()
        client._stream_active = True
        client._on_websocket_message({"fresh": "message"})
        msg = client._message_queue.get(timeout=1)
        assert msg == {"fresh": "message"}


class TestStreamMessageControl:
    """Tests for stream message control flow."""

    def test_stream_messages_stop_event_breaks(self, monkeypatch: pytest.MonkeyPatch):
        client = MockWebSocketClient()
        stop_event = threading.Event()
        stop_event.set()
        client._stream_active = True

        monkeypatch.setattr(websocket_mixin_module, "_QUEUE_TIMEOUT", 0.0)
        messages = list(client._stream_messages(stop_event))

        assert messages == []

    def test_stream_messages_stop_sentinel_breaks(self):
        client = MockWebSocketClient()
        client._stream_active = True
        client._message_queue.put(_STREAM_STOP)

        messages = list(client._stream_messages())

        assert messages == []
