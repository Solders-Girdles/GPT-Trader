"""Unit tests for WebSocketClientMixin trade streaming and stream control."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import _STREAM_STOP
from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
)


class TestStreamTrades:
    """Tests for stream_trades method."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_connects_and_subscribes_to_market_trades(self, mock_ws_class):
        """stream_trades should subscribe to market_trades channel."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_trades(["BTC-USD", "ETH-USD"], stop_event=stop_event):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.connect.assert_called_once()
        mock_ws.subscribe.assert_called_once_with(["BTC-USD", "ETH-USD"], ["market_trades"])

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_yields_trade_messages(self, mock_ws_class):
        """Trade messages should be yielded via the queue mechanism."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Directly test the message callback and queue mechanism
        client._stream_active = True

        client._on_websocket_message(
            {"type": "market_trades", "product_id": "BTC-USD", "price": "49999", "size": "0.1"}
        )

        # Verify message is in queue
        msg = client._message_queue.get(timeout=1)

        assert msg["price"] == "49999"
        assert msg["size"] == "0.1"


class TestStreamControl:
    """Tests for stream control methods."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_stop_streaming_disconnects_websocket(self, mock_ws_class):
        """stop_streaming should disconnect WebSocket and reset state."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Start a stream
        stop_event = threading.Event()
        subscribed = threading.Event()
        mock_ws.subscribe.side_effect = lambda *args, **kwargs: subscribed.set()

        def consume_stream():
            for _ in client.stream_orderbook(["BTC-USD"], stop_event=stop_event):
                pass

        thread = threading.Thread(target=consume_stream)
        thread.start()

        assert subscribed.wait(timeout=1.0)

        # Stop streaming
        client.stop_streaming()
        thread.join(timeout=2)
        assert not thread.is_alive()

        mock_ws.disconnect.assert_called_once()
        assert client._ws is None
        assert client._stream_active is False

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_stop_streaming_handles_disconnect_error(self, mock_ws_class):
        """stop_streaming should handle disconnect errors and clear state."""
        mock_ws = MagicMock()
        mock_ws.disconnect.side_effect = RuntimeError("disconnect failed")
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()
        client._get_websocket()

        client.stop_streaming()

        assert client._ws is None
        assert client._stream_active is False

    def test_is_streaming_returns_correct_state(self):
        """is_streaming should reflect _stream_active flag."""
        client = MockWebSocketClient()

        # Initially not streaming
        assert client.is_streaming() is False

        # Manually set stream active
        client._stream_active = True
        assert client.is_streaming() is True

        # Manually set stream inactive
        client._stream_active = False
        assert client.is_streaming() is False


class TestStreamMessageControl:
    """Tests for stream message control flow."""

    def test_stream_messages_stop_event_breaks(self):
        client = MockWebSocketClient()
        stop_event = threading.Event()
        stop_event.set()
        client._stream_active = True

        with patch(
            "gpt_trader.features.brokerages.coinbase.client.websocket_mixin._QUEUE_TIMEOUT",
            0.0,
        ):
            messages = list(client._stream_messages(stop_event))

        assert messages == []

    def test_stream_messages_stop_sentinel_breaks(self):
        client = MockWebSocketClient()
        client._stream_active = True
        client._message_queue.put(_STREAM_STOP)

        messages = list(client._stream_messages())

        assert messages == []
