"""Unit tests for WebSocketClientMixin initialization and lifecycle."""

from __future__ import annotations

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

        # Should only create once
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
        self,
        mock_websocket_class: MagicMock,
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
