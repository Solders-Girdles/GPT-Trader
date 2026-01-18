"""Focused RealTransport coverage tests."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from gpt_trader.features.brokerages.coinbase.transports import RealTransport


class TestRealTransportCoverage:
    """RealTransport functionality tests for maximum coverage impact."""

    def test_real_transport_initialization_default(self):
        """Test RealTransport initialization with default settings."""
        transport = RealTransport()

        assert transport.ws is None
        assert transport.url is None
        assert transport._config is None

    def test_real_transport_initialization_with_config(self, mock_bot_config):
        """Test RealTransport initialization with custom config."""
        transport = RealTransport(config=mock_bot_config)

        assert transport._config is mock_bot_config

    def test_connect_success(self, mock_bot_config):
        """Test successful WebSocket connection."""
        transport = RealTransport(config=mock_bot_config)

        with patch("websocket.create_connection") as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            url = "wss://test.example.com"
            headers = {"Authorization": "Bearer token"}
            transport.connect(url, headers)

            assert transport.url == url
            assert transport.ws is mock_ws
            mock_create.assert_called_once_with(url, header=headers)

    def test_connect_with_headers_dict_uses_existing_url(self, mock_bot_config):
        """Test connection when headers passed as first argument."""
        transport = RealTransport(config=mock_bot_config)
        transport.url = "wss://test.example.com"

        with patch("websocket.create_connection") as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws
            headers = {"Authorization": "Bearer token"}

            transport.connect(headers)

            mock_create.assert_called_once_with("wss://test.example.com", header=headers)
            assert transport.ws is mock_ws

    def test_connect_headers_without_url_raises(self, mock_bot_config):
        """Test connection fails when url is missing and headers passed as first argument."""
        transport = RealTransport(config=mock_bot_config)

        with pytest.raises(ValueError, match="WebSocket URL is required"):
            transport.connect({"Authorization": "Bearer token"})

    def test_connect_with_environment_options(self, mock_bot_config, monkeypatch):
        """Test connection with environment-based options."""
        monkeypatch.setenv("COINBASE_WS_CONNECT_TIMEOUT", "30.0")
        monkeypatch.setenv("COINBASE_WS_SUBPROTOCOLS", "v1,v2")
        monkeypatch.setenv("COINBASE_WS_ENABLE_TRACE", "true")

        transport = RealTransport(config=mock_bot_config)

        with (
            patch("websocket.create_connection") as mock_create,
            patch("websocket.enableTrace") as mock_trace,
        ):
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            url = "wss://test.example.com"
            transport.connect(url)

            expected_options = {
                "timeout": 30.0,
                "subprotocols": ["v1", "v2"],
            }
            mock_create.assert_called_once_with(url, **expected_options)
            mock_trace.assert_called_once_with(True)

    def test_connect_invalid_timeout(self, mock_bot_config, monkeypatch, caplog):
        """Test connection with invalid timeout value."""
        monkeypatch.setenv("COINBASE_WS_CONNECT_TIMEOUT", "invalid")

        transport = RealTransport(config=mock_bot_config)

        with patch("websocket.create_connection") as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            url = "wss://test.example.com"
            transport.connect(url)

            assert "Ignoring invalid COINBASE_WS_CONNECT_TIMEOUT" in caplog.text
            mock_create.assert_called_once_with(url)

    def test_connect_missing_websocket_client(self, mock_bot_config):
        """Test connection when websocket-client is not installed."""
        transport = RealTransport(config=mock_bot_config)

        with patch(
            "gpt_trader.features.brokerages.coinbase.transports.websocket.create_connection",
            side_effect=ModuleNotFoundError("No module named 'websocket'"),
        ):
            with pytest.raises(ImportError, match="websocket-client is not installed"):
                transport.connect("wss://test.example.com")

    def test_connect_trace_enable_failure(self, mock_bot_config, monkeypatch, caplog):
        """Test connection when trace enable fails."""
        monkeypatch.setenv("COINBASE_WS_ENABLE_TRACE", "true")

        transport = RealTransport(config=mock_bot_config)

        with (
            patch("websocket.create_connection") as mock_create,
            patch("websocket.enableTrace", side_effect=Exception("Trace failed")),
        ):
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            transport.connect("wss://test.example.com")

            assert "Unable to enable websocket trace output" in caplog.text

    def test_disconnect_success(self, mock_bot_config):
        """Test successful WebSocket disconnection."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        transport.disconnect()

        mock_ws.close.assert_called_once()
        assert transport.ws is None

    def test_disconnect_with_error(self, mock_bot_config, caplog):
        """Test disconnection when close raises an error."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        mock_ws.close.side_effect = Exception("Close failed")
        transport.ws = mock_ws

        transport.disconnect()

        assert "Error disconnecting" in caplog.text
        assert transport.ws is None

    def test_disconnect_no_websocket(self, mock_bot_config):
        """Test disconnection when no WebSocket is connected."""
        transport = RealTransport(config=mock_bot_config)

        transport.disconnect()

        assert transport.ws is None

    def test_subscribe_success(self, mock_bot_config):
        """Test successful subscription message sending."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        message = {"type": "subscribe", "channels": ["ticker"]}
        transport.subscribe(message)

        mock_ws.send.assert_called_once_with(json.dumps(message))

    def test_subscribe_not_connected(self, mock_bot_config):
        """Test subscription when not connected."""
        transport = RealTransport(config=mock_bot_config)

        message = {"type": "subscribe", "channels": ["ticker"]}

        with pytest.raises(RuntimeError, match="Not connected to WebSocket"):
            transport.subscribe(message)

    def test_stream_success(self, mock_bot_config):
        """Test successful message streaming."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        messages = ['{"type": "ticker", "price": "50000"}', '{"type": "match", "size": "0.1"}']
        mock_ws.recv.side_effect = messages

        stream = transport.stream()
        result = list(stream)

        expected = [
            {"type": "ticker", "price": "50000"},
            {"type": "match", "size": "0.1"},
        ]
        assert result == expected

    def test_stream_handles_stop_iteration(self, mock_bot_config):
        """Test streaming stops on StopIteration from websocket."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws
        mock_ws.recv.side_effect = StopIteration

        result = list(transport.stream())

        assert result == []

    def test_stream_not_connected(self, mock_bot_config):
        """Test streaming when not connected."""
        transport = RealTransport(config=mock_bot_config)

        with pytest.raises(RuntimeError, match="Not connected to WebSocket"):
            list(transport.stream())

    def test_stream_with_error(self, mock_bot_config, caplog):
        """Test streaming when WebSocket recv raises an error."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws
        mock_ws.recv.side_effect = Exception("Connection lost")

        with pytest.raises(Exception, match="Connection lost"):
            list(transport.stream())

        assert "WebSocket stream error" in caplog.text
