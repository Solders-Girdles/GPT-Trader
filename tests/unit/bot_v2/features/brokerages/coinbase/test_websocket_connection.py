"""
WebSocket connection lifecycle and transport tests.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.transports import (
    MockTransport,
)
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
)


class TestCoinbaseWebSocketConnection:
    """Test WebSocket connection lifecycle and management."""

    def test_websocket_initialization_with_settings(self, mock_runtime_settings):
        """Test WebSocket initialization with runtime settings."""
        ws = CoinbaseWebSocket(
            url="wss://ws-feed.exchange.coinbase.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        assert ws.url == "wss://ws-feed.exchange.coinbase.com"
        assert ws._settings is mock_runtime_settings
        assert ws._ws_auth_provider is None
        assert ws._transport is None
        assert ws._sub is None

    def test_websocket_initialization_without_settings(self):
        """Test WebSocket initialization without runtime settings."""
        ws = CoinbaseWebSocket(
            url="wss://ws-feed.exchange.coinbase.com",
            ws_auth_provider=None,
        )

        assert ws._url == "wss://ws-feed.exchange.coinbase.com"
        assert ws._settings is not None  # Should load default settings
        assert ws._auth_provider is None
        assert ws._transport is None

    def test_connect_with_mock_transport(self, mock_runtime_settings):
        """Test connecting with mock transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Mock the transport creation
        with patch.object(ws, "_create_transport") as mock_create:
            mock_transport = MockTransport()
            mock_create.return_value = mock_transport

            ws.connect()

            mock_create.assert_called_once()
            mock_transport.connect.assert_called_once_with("wss://test.example.com", None)
            assert ws._transport is mock_transport

    def test_connect_with_auth_headers(self, mock_runtime_settings):
        """Test connecting with authentication headers."""
        auth_provider = Mock()
        auth_provider.get_headers.return_value = {"Authorization": "Bearer token123"}

        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=auth_provider,
            settings=mock_runtime_settings,
        )

        with patch.object(ws, "_create_transport") as mock_create:
            mock_transport = MockTransport()
            mock_create.return_value = mock_transport

            ws.connect()

            expected_headers = {"Authorization": "Bearer token123"}
            mock_transport.connect.assert_called_once_with(
                "wss://test.example.com", expected_headers
            )

    def test_connect_already_connected(self, mock_runtime_settings):
        """Test connecting when already connected."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup existing transport
        mock_transport = MockTransport()
        ws._transport = mock_transport

        # Try to connect again
        ws.connect()

        # Should not create new transport
        assert ws._transport is mock_transport

    def test_disconnect_with_transport(self, mock_runtime_settings):
        """Test disconnecting with active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup transport
        mock_transport = MockTransport()
        ws._transport = mock_transport

        ws.disconnect()

        mock_transport.disconnect.assert_called_once()
        assert ws._transport is None

    def test_disconnect_without_transport(self, mock_runtime_settings):
        """Test disconnecting without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Should not raise exception
        ws.disconnect()
        assert ws._transport is None

    def test_is_connected_true(self, mock_runtime_settings):
        """Test is_connected returns True when transport is connected."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        mock_transport.connected = True
        ws._transport = mock_transport

        assert ws.is_connected() is True

    def test_is_connected_false(self, mock_runtime_settings):
        """Test is_connected returns False when transport is not connected."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # No transport
        assert ws.is_connected() is False

        # Transport not connected
        mock_transport = MockTransport()
        mock_transport.connected = False
        ws._transport = mock_transport

        assert ws.is_connected() is False


class TestCoinbaseWebSocketTransportCreation:
    """Test WebSocket transport creation and configuration."""

    def test_create_real_transport_by_default(self, mock_runtime_settings):
        """Test that RealTransport is created by default."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport = Mock()
            mock_real.return_value = mock_transport

            transport = ws._create_transport()

            mock_real.assert_called_once_with(settings=mock_runtime_settings)
            assert transport is mock_transport

    def test_create_mock_transport_when_websocket_client_unavailable(self, mock_runtime_settings):
        """Test creating MockTransport when websocket-client is unavailable."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            # Simulate ImportError
            mock_real.side_effect = ImportError("websocket-client not available")

            with patch("bot_v2.features.brokerages.coinbase.ws.MockTransport") as mock_mock:
                mock_transport = Mock()
                mock_mock.return_value = mock_transport

                transport = ws._create_transport()

                mock_mock.assert_called_once()
                assert transport is mock_transport

    def test_create_noop_transport_when_streaming_disabled(self, mock_runtime_settings):
        """Test creating NoopTransport when streaming is disabled."""
        # Set environment variable to disable streaming
        mock_runtime_settings.raw_env["COINBASE_WS_DISABLE_STREAMING"] = "true"

        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.NoopTransport") as mock_noop:
            mock_transport = Mock()
            mock_noop.return_value = mock_transport

            transport = ws._create_transport()

            mock_noop.assert_called_once_with(settings=mock_runtime_settings)
            assert transport is mock_transport

    def test_transport_settings_integration(self, mock_runtime_settings):
        """Test that runtime settings are properly passed to transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport = Mock()
            mock_real.return_value = mock_transport

            ws._create_transport()

            # Verify settings were passed
            mock_real.assert_called_once_with(settings=mock_runtime_settings)


class TestWebSocketConnectionErrorHandling:
    """Test WebSocket connection error handling and recovery."""

    def test_connection_error_handling(self, mock_runtime_settings, connection_error_scenarios):
        """Test handling of various connection errors."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        for error_name, error in connection_error_scenarios.items():
            with patch.object(ws, "_create_transport") as mock_create:
                mock_create.side_effect = error

                with pytest.raises(type(error)):
                    ws.connect()

    def test_transport_error_propagation(self, mock_runtime_settings):
        """Test that transport errors are properly propagated."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        mock_transport.connect.side_effect = ConnectionError("Connection failed")
        ws._transport = mock_transport

        with pytest.raises(ConnectionError):
            ws.connect()

    def test_graceful_degradation_on_missing_features(self, mock_runtime_settings):
        """Test graceful degradation when optional features are unavailable."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Should work even without authentication
        ws._auth_provider = None
        ws.connect()  # Should not raise exception

    def test_reconnection_scenario(self, mock_runtime_settings):
        """Test reconnection scenario with backoff."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        connect_attempts = []

        def mock_connect(*args, **kwargs):
            connect_attempts.append(len(connect_attempts) + 1)
            if len(connect_attempts) <= 2:
                raise ConnectionError(f"Attempt {len(connect_attempts)} failed")

        with patch.object(ws, "_create_transport") as mock_create:
            mock_transport = MockTransport()
            mock_transport.connect = mock_connect
            mock_create.return_value = mock_transport

            # First connection attempt
            try:
                ws.connect()
            except ConnectionError:
                pass

            # Second connection attempt
            try:
                ws.connect()
            except ConnectionError:
                pass

            # Third connection should succeed
            ws.connect()

            assert len(connect_attempts) == 3
            assert ws.is_connected()
