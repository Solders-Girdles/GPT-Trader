"""Unit tests for WebSocketClientMixin stream_orderbook."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client.websocket_mixin as websocket_mixin_module
from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
)


@pytest.fixture
def mock_websocket_class(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_ws_class = MagicMock()
    monkeypatch.setattr(websocket_mixin_module, "CoinbaseWebSocket", mock_ws_class)
    return mock_ws_class


class TestStreamOrderbook:
    """Tests for stream_orderbook method."""

    def test_connects_and_subscribes_to_level2(self, mock_websocket_class: MagicMock):
        """stream_orderbook with level>=2 should subscribe to level2 channel."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Start stream in background thread
        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(
                ["BTC-USD", "ETH-USD"], level=2, stop_event=stop_event
            ):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.connect.assert_called_once()
        mock_ws.subscribe.assert_called_once_with(["BTC-USD", "ETH-USD"], ["level2"])

    def test_subscribes_to_ticker_for_level1(self, mock_websocket_class: MagicMock):
        """stream_orderbook with level=1 should subscribe to ticker channel."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()

        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(["BTC-USD"], level=1, stop_event=stop_event):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.subscribe.assert_called_once_with(["BTC-USD"], ["ticker"])

    def test_include_trades_adds_channel(self, mock_websocket_class: MagicMock):
        """stream_orderbook should include market_trades when requested."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()
        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(
                ["BTC-USD"], level=2, stop_event=stop_event, include_trades=True
            ):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.subscribe.assert_called_once_with(["BTC-USD"], ["level2", "market_trades"])

    def test_include_user_events_subscribes(self, mock_websocket_class: MagicMock):
        """stream_orderbook should subscribe to user events when requested."""
        mock_ws = MagicMock()
        mock_websocket_class.return_value = mock_ws

        client = MockWebSocketClient()
        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(
                ["BTC-USD"], level=1, stop_event=stop_event, include_user_events=True
            ):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.subscribe_user_events.assert_called_once_with(["BTC-USD"])

    def test_yields_messages_from_callback(self):
        """Messages pushed via callback should be yielded by stream."""
        client = MockWebSocketClient()

        # Directly test the message callback and queue mechanism
        # Set stream_active manually to simulate an active stream
        client._stream_active = True

        # Push messages via callback
        client._on_websocket_message({"type": "level2", "product_id": "BTC-USD", "price": "50000"})
        client._on_websocket_message({"type": "level2", "product_id": "BTC-USD", "price": "50001"})

        # Verify messages are in queue
        msg1 = client._message_queue.get(timeout=1)
        msg2 = client._message_queue.get(timeout=1)

        assert msg1["price"] == "50000"
        assert msg2["price"] == "50001"
