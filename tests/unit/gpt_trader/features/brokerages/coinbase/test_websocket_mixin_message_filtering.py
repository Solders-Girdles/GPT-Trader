"""Unit tests for WebSocketClientMixin message filtering."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client.websocket_mixin as ws_mixin_module
from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
)


class TestMessageFiltering:
    """Tests for message filtering behavior."""

    def test_ignores_non_dict_messages(self, monkeypatch: pytest.MonkeyPatch):
        """Non-dict messages should be ignored."""
        mock_ws = MagicMock()
        mock_ws_class = MagicMock(return_value=mock_ws)
        monkeypatch.setattr(ws_mixin_module, "CoinbaseWebSocket", mock_ws_class)

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

        # Push non-dict messages (these would be filtered by _stream_messages)
        client._message_queue.put("string message")
        client._message_queue.put(123)
        client._message_queue.put({"valid": "message"})

        assert received_event.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        # Only dict message should be yielded
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

        # Pre-populate queue with stale messages
        client._message_queue.put({"stale": "message1"})
        client._message_queue.put({"stale": "message2"})

        # Verify queue has messages
        assert not client._message_queue.empty()

        # Clear the queue
        client._clear_queue()

        # Verify queue is now empty
        assert client._message_queue.empty()

        # Push fresh message and verify it's received
        client._stream_active = True
        client._on_websocket_message({"fresh": "message"})

        msg = client._message_queue.get(timeout=1)
        assert msg == {"fresh": "message"}
