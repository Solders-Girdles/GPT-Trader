"""Unit tests for WebSocketClientMixin initialization, lifecycle, and health."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client.websocket_mixin as websocket_mixin_module
from tests.unit.gpt_trader.features.brokerages.coinbase.websocket_mixin_test_helpers import (
    MockWebSocketClient,
    _NonCooperativeClient,
)


@pytest.fixture
def mock_websocket_class(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_ws_class = MagicMock()
    monkeypatch.setattr(websocket_mixin_module, "CoinbaseWebSocket", mock_ws_class)
    return mock_ws_class


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

    def test_creates_websocket_on_first_call(self, mock_websocket_class: MagicMock):
        """First call should create a new WebSocket instance."""
        mock_ws_instance = MagicMock()
        mock_websocket_class.return_value = mock_ws_instance
        client = MockWebSocketClient()
        result = client._get_websocket()
        mock_websocket_class.assert_called_once()
        assert result is mock_ws_instance
        assert client._ws is mock_ws_instance

    def test_returns_same_websocket_on_subsequent_calls(self, mock_websocket_class: MagicMock):
        """Subsequent calls should return the same WebSocket instance."""
        mock_ws_instance = MagicMock()
        mock_websocket_class.return_value = mock_ws_instance
        client = MockWebSocketClient()
        first = client._get_websocket()
        second = client._get_websocket()
        mock_websocket_class.assert_called_once()
        assert first is second

    def test_passes_auth_credentials_to_websocket(self, mock_websocket_class: MagicMock):
        """WebSocket should receive auth credentials if available."""
        mock_auth = MagicMock()
        mock_auth.api_key = "my_api_key"
        mock_auth.private_key = "my_private_key"
        client = MockWebSocketClient(auth=mock_auth)
        client._get_websocket()
        mock_websocket_class.assert_called_once()
        call_kwargs = mock_websocket_class.call_args.kwargs
        assert call_kwargs["api_key"] == "my_api_key"
        assert call_kwargs["private_key"] == "my_private_key"

    def test_handles_no_auth(self, mock_websocket_class: MagicMock):
        """WebSocket should work without auth credentials."""
        client = MockWebSocketClient(auth=None)
        client._get_websocket()
        mock_websocket_class.assert_called_once()
        call_kwargs = mock_websocket_class.call_args.kwargs
        assert call_kwargs["api_key"] is None
        assert call_kwargs["private_key"] is None


class TestWebSocketClientMixinNonCooperativeInit:
    def test_lazy_initializes_state_when_mixin_init_not_called(
        self, mock_websocket_class: MagicMock
    ) -> None:
        client = _NonCooperativeClient()
        assert not hasattr(client, "_ws_lock")
        assert not hasattr(client, "_message_queue")
        mock_ws_instance = MagicMock()
        mock_websocket_class.return_value = mock_ws_instance
        result = client._get_websocket()
        assert result is mock_ws_instance
        assert client._ws is mock_ws_instance
        assert client._ws_lock is not None
        assert client._message_queue is not None

    def test_stop_streaming_safe_when_mixin_init_not_called(self) -> None:
        client = _NonCooperativeClient()
        client.stop_streaming()


class TestGetWSHealth:
    """Tests for get_ws_health method."""

    def test_returns_empty_dict_when_no_ws(self, mock_websocket_class: MagicMock):
        """get_ws_health should return empty dict when no WebSocket exists."""
        client = MockWebSocketClient()
        assert client._ws is None
        result = client.get_ws_health()
        assert result == {}

    def test_returns_health_dict_from_ws(self, mock_websocket_class: MagicMock):
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
        mock_websocket_class.return_value = mock_ws
        client = MockWebSocketClient()
        client._get_websocket()
        result = client.get_ws_health()
        assert result == mock_health
        mock_ws.get_health.assert_called_once()

    def test_returns_empty_dict_after_stop_streaming(self, mock_websocket_class: MagicMock):
        """get_ws_health should return empty dict after WebSocket is cleaned up."""
        mock_ws = MagicMock()
        mock_ws.get_health.return_value = {"connected": True}
        mock_websocket_class.return_value = mock_ws
        client = MockWebSocketClient()
        client._get_websocket()
        client.stop_streaming()
        result = client.get_ws_health()
        assert result == {}
        assert client._ws is None

    def test_is_thread_safe(self, mock_websocket_class: MagicMock):
        """get_ws_health should be thread-safe using the lock."""
        mock_ws = MagicMock()
        mock_ws.get_health.return_value = {"connected": True}
        mock_websocket_class.return_value = mock_ws
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
        assert len(results) == 50
        assert all(r == {"connected": True} for r in results)
