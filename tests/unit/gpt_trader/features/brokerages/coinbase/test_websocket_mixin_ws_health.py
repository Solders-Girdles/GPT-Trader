"""Unit tests for WebSocketClientMixin get_ws_health."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
)


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
