"""
Unit tests for WebSocketClientMixin.

Tests the stream_orderbook and stream_trades methods that bridge
CoinbaseWebSocket with the broker interface.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import (
    _STREAM_STOP,
    WebSocketClientMixin,
)


class MockClientBase:
    """Mock base class providing auth attribute."""

    def __init__(self, auth: MagicMock | None = None) -> None:
        self.auth = auth


class MockWebSocketClient(WebSocketClientMixin, MockClientBase):
    """Testable client combining mixin with mock base."""

    def __init__(self, auth: MagicMock | None = None) -> None:
        super().__init__(auth=auth)


class TestWebSocketClientMixinInitialization:
    """Tests for mixin initialization."""

    def test_initialization_creates_attributes(self):
        """Mixin should initialize WebSocket-related attributes."""
        client = MockWebSocketClient()

        assert client._ws is None
        assert client._ws_lock is not None
        assert client._message_queue is not None
        assert client._stream_active is False

    def test_initialization_with_auth(self):
        """Mixin should accept auth from base class."""
        mock_auth = MagicMock()
        mock_auth.api_key = "test_key"
        mock_auth.private_key = "test_secret"

        client = MockWebSocketClient(auth=mock_auth)

        assert client.auth is mock_auth


class TestWebSocketClientMixinGetWebSocket:
    """Tests for _get_websocket singleton management."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_creates_websocket_on_first_call(self, mock_ws_class):
        """First call should create a new WebSocket instance."""
        mock_ws_instance = MagicMock()
        mock_ws_class.return_value = mock_ws_instance

        client = MockWebSocketClient()
        result = client._get_websocket()

        mock_ws_class.assert_called_once()
        assert result is mock_ws_instance
        assert client._ws is mock_ws_instance

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_returns_same_websocket_on_subsequent_calls(self, mock_ws_class):
        """Subsequent calls should return the same WebSocket instance."""
        mock_ws_instance = MagicMock()
        mock_ws_class.return_value = mock_ws_instance

        client = MockWebSocketClient()

        first = client._get_websocket()
        second = client._get_websocket()

        # Should only create once
        mock_ws_class.assert_called_once()
        assert first is second

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_passes_auth_credentials_to_websocket(self, mock_ws_class):
        """WebSocket should receive auth credentials if available."""
        mock_auth = MagicMock()
        mock_auth.api_key = "my_api_key"
        mock_auth.private_key = "my_private_key"

        client = MockWebSocketClient(auth=mock_auth)
        client._get_websocket()

        mock_ws_class.assert_called_once()
        call_kwargs = mock_ws_class.call_args.kwargs
        assert call_kwargs["api_key"] == "my_api_key"
        assert call_kwargs["private_key"] == "my_private_key"

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_handles_no_auth(self, mock_ws_class):
        """WebSocket should work without auth credentials."""
        client = MockWebSocketClient(auth=None)
        client._get_websocket()

        mock_ws_class.assert_called_once()
        call_kwargs = mock_ws_class.call_args.kwargs
        assert call_kwargs["api_key"] is None
        assert call_kwargs["private_key"] is None


class TestStreamOrderbook:
    """Tests for stream_orderbook method."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_connects_and_subscribes_to_level2(self, mock_ws_class):
        """stream_orderbook with level>=2 should subscribe to level2 channel."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

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

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_subscribes_to_ticker_for_level1(self, mock_ws_class):
        """stream_orderbook with level=1 should subscribe to ticker channel."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

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

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_include_trades_adds_channel(self, mock_ws_class):
        """stream_orderbook should include market_trades when requested."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

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

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_yields_messages_from_callback(self, mock_ws_class):
        """Messages pushed via callback should be yielded by stream."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

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

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_is_streaming_returns_correct_state(self, mock_ws_class):
        """is_streaming should reflect _stream_active flag."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Initially not streaming
        assert client.is_streaming() is False

        # Manually set stream active
        client._stream_active = True
        assert client.is_streaming() is True

        # Manually set stream inactive
        client._stream_active = False
        assert client.is_streaming() is False


class TestMessageFiltering:
    """Tests for message filtering behavior."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_ignores_non_dict_messages(self, mock_ws_class):
        """Non-dict messages should be ignored."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()
        received_messages = []
        stop_event = threading.Event()
        received_event = threading.Event()

        def consume_stream():
            for msg in client.stream_orderbook(["BTC-USD"], stop_event=stop_event):
                received_messages.append(msg)
                received_event.set()

        thread = threading.Thread(target=consume_stream)
        thread.start()

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

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_clears_queue_before_streaming(self, mock_ws_class):
        """_clear_queue should remove all pending messages."""
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws

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


class TestGetWSHealth:
    """Tests for get_ws_health method."""

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_returns_empty_dict_when_no_ws(self, mock_ws_class):
        """get_ws_health should return empty dict when no WebSocket exists."""
        client = MockWebSocketClient()

        # No WebSocket created yet
        assert client._ws is None
        result = client.get_ws_health()

        assert result == {}

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_returns_health_dict_from_ws(self, mock_ws_class):
        """get_ws_health should return health dict from underlying WebSocket."""
        mock_ws = MagicMock()
        mock_health = {
            "connected": True,
            "last_message_ts": 1234567890.0,
            "last_heartbeat_ts": 1234567890.0,
            "last_close_ts": None,
            "last_error_ts": None,
            "gap_count": 2,
            "reconnect_count": 1,
        }
        mock_ws.get_health.return_value = mock_health
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Create the WebSocket
        client._get_websocket()

        result = client.get_ws_health()

        assert result == mock_health
        mock_ws.get_health.assert_called_once()

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_returns_empty_dict_after_stop_streaming(self, mock_ws_class):
        """get_ws_health should return empty dict after WebSocket is cleaned up."""
        mock_ws = MagicMock()
        mock_ws.get_health.return_value = {"connected": True}
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()

        # Create and then stop
        client._get_websocket()
        client.stop_streaming()

        result = client.get_ws_health()

        assert result == {}
        assert client._ws is None

    @patch("gpt_trader.features.brokerages.coinbase.client.websocket_mixin.CoinbaseWebSocket")
    def test_is_thread_safe(self, mock_ws_class):
        """get_ws_health should be thread-safe using the lock."""
        mock_ws = MagicMock()
        mock_ws.get_health.return_value = {"connected": True}
        mock_ws_class.return_value = mock_ws

        client = MockWebSocketClient()
        client._get_websocket()

        results = []

        def call_get_health():
            for _ in range(10):
                result = client.get_ws_health()
                results.append(result)

        threads = [threading.Thread(target=call_get_health) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All calls should have succeeded
        assert len(results) == 50
        assert all(r == {"connected": True} for r in results)


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
