"""
Focused transport coverage tests for 80%+ coverage improvement.

This test suite targets all three transport classes: RealTransport, MockTransport, and NoopTransport
which are critical for WebSocket connection management and testing infrastructure.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from gpt_trader.features.brokerages.coinbase.transports import (
    MockTransport,
    NoopTransport,
    RealTransport,
)


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

    def test_connect_with_environment_options(self, mock_bot_config, monkeypatch):
        """Test connection with environment-based options."""
        # Set environment variables via monkeypatch
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

            # Should log warning about invalid timeout
            assert "Ignoring invalid COINBASE_WS_CONNECT_TIMEOUT" in caplog.text
            mock_create.assert_called_once_with(url)

    def test_connect_missing_websocket_client(self, mock_bot_config):
        """Test connection when websocket-client is not installed."""
        transport = RealTransport(config=mock_bot_config)

        # We patch create_connection on the imported websocket module in transports.py
        # to simulate the module being missing/broken at runtime
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

            # Should log warning message about trace failure
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

        # Should log error but still set ws to None
        assert "Error disconnecting" in caplog.text
        assert transport.ws is None

    def test_disconnect_no_websocket(self, mock_bot_config):
        """Test disconnection when no WebSocket is connected."""
        transport = RealTransport(config=mock_bot_config)

        # Should not raise exception
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


class TestMockTransportCoverage:
    """MockTransport functionality tests for maximum coverage impact."""

    def test_mock_transport_initialization_default(self):
        """Test MockTransport initialization with default empty messages."""
        transport = MockTransport()

        assert transport.messages == []
        assert transport.connected is False
        assert transport.subscriptions == []

    def test_mock_transport_initialization_with_messages(self):
        """Test MockTransport initialization with predefined messages."""
        messages = [{"type": "ticker"}, {"type": "match"}]
        transport = MockTransport(messages=messages)

        assert transport.messages == messages
        assert transport.connected is False
        assert transport.subscriptions == []

    def test_mock_transport_connect(self):
        """Test MockTransport connection."""
        transport = MockTransport()
        headers = {"Authorization": "Bearer token"}

        transport.connect("wss://test.example.com", headers)

        assert transport.connected is True
        assert transport.headers == headers

    def test_mock_transport_connect_without_headers(self):
        """Test MockTransport connection without headers."""
        transport = MockTransport()

        transport.connect("wss://test.example.com")

        assert transport.connected is True
        # MockTransport stores headers as None when not provided
        assert transport.headers is None

    def test_mock_transport_disconnect(self):
        """Test MockTransport disconnection."""
        transport = MockTransport()
        transport.connected = True

        transport.disconnect()

        assert transport.connected is False

    def test_mock_transport_subscribe(self):
        """Test MockTransport subscription recording."""
        transport = MockTransport()
        message = {"type": "subscribe", "channels": ["ticker"]}

        transport.subscribe(message)

        assert len(transport.subscriptions) == 1
        assert transport.subscriptions[0] == message

    def test_mock_transport_stream(self):
        """Test MockTransport message streaming."""
        messages = [{"type": "ticker"}, {"type": "match"}]
        transport = MockTransport(messages=messages)

        result = list(transport.stream())

        assert result == messages

    def test_mock_transport_add_message(self):
        """Test adding messages to MockTransport."""
        transport = MockTransport()
        message = {"type": "ticker", "price": "50000"}

        transport.add_message(message)

        assert len(transport.messages) == 1
        assert transport.messages[0] == message


class TestNoopTransportCoverage:
    """NoopTransport functionality tests for maximum coverage impact."""

    def test_noop_transport_initialization_default(self):
        """Test NoopTransport initialization with default settings."""
        transport = NoopTransport()

        assert transport.connected is False
        assert transport._config is None

    def test_noop_transport_initialization_with_config(self, mock_bot_config):
        """Test NoopTransport initialization with custom config."""
        transport = NoopTransport(config=mock_bot_config)

        assert transport._config is mock_bot_config

    def test_noop_transport_connect(self):
        """Test NoopTransport connection."""
        transport = NoopTransport()

        transport.connect("wss://test.example.com")

        assert transport.connected is True

    def test_noop_transport_connect_with_headers(self):
        """Test NoopTransport connection with headers (should be ignored)."""
        transport = NoopTransport()
        headers = {"Authorization": "Bearer token"}

        transport.connect("wss://test.example.com", headers)

        assert transport.connected is True

    def test_noop_transport_disconnect(self):
        """Test NoopTransport disconnection."""
        transport = NoopTransport()
        transport.connected = True

        transport.disconnect()

        assert transport.connected is False

    def test_noop_transport_subscribe(self):
        """Test NoopTransport subscription (should be ignored)."""
        transport = NoopTransport()
        message = {"type": "subscribe", "channels": ["ticker"]}

        transport.subscribe(message)
        assert transport.connected is False

    def test_noop_transport_stream(self):
        """Test NoopTransport streaming (should return empty iterator)."""
        transport = NoopTransport()

        result = list(transport.stream())

        assert result == []


class TestTransportEdgeCases:
    """Test edge cases and error scenarios."""

    def test_real_transport_malformed_json_stream(self, mock_bot_config):
        """Test streaming malformed JSON messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = '{"invalid": json structure}'

        with pytest.raises(json.JSONDecodeError):
            list(transport.stream())

    def test_real_transport_empty_message_stream(self, mock_bot_config):
        """Test streaming empty messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = ""

        # Empty string should be decoded to JSON with error
        with pytest.raises(json.JSONDecodeError):
            list(transport.stream())

    def test_real_transport_none_message_stream(self, mock_bot_config):
        """Test streaming None messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = None

        # None should cause JSON decoding error
        with pytest.raises(TypeError):
            list(transport.stream())

    def test_mock_transport_multiple_subscriptions(self):
        """Test MockTransport with multiple subscriptions."""
        transport = MockTransport()
        messages = [
            {"type": "subscribe", "channels": ["ticker"]},
            {"type": "subscribe", "channels": ["matches"]},
            {"type": "subscribe", "channels": ["level2"]},
        ]

        for message in messages:
            transport.subscribe(message)

        assert len(transport.subscriptions) == 3
        assert transport.subscriptions == messages

    def test_transport_environment_variable_parsing(self, mock_bot_config, monkeypatch):
        """Test various environment variable parsing scenarios."""
        # Test various truthy values
        for truthy_value in ["1", "true", "yes", "on", "TRUE", "Yes", "ON"]:
            monkeypatch.setenv("COINBASE_WS_ENABLE_TRACE", truthy_value)

            transport = RealTransport(config=mock_bot_config)

            with (
                patch("websocket.create_connection") as mock_create,
                patch("websocket.enableTrace") as mock_trace,
            ):
                mock_ws = Mock()
                mock_create.return_value = mock_ws
                transport.connect("wss://test.example.com")

                mock_trace.assert_called_once_with(True)

    def test_transport_empty_subprotocols(self, mock_bot_config, monkeypatch):
        """Test empty subprotocols configuration."""
        monkeypatch.setenv("COINBASE_WS_SUBPROTOCOLS", "   ,  , ")

        transport = RealTransport(config=mock_bot_config)

        with patch("websocket.create_connection") as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            transport.connect("wss://test.example.com")

            # Should not include subprotocols in options
            call_args = mock_create.call_args[1]
            assert "subprotocols" not in call_args

    def test_transport_mixed_whitespace_subprotocols(self, mock_bot_config, monkeypatch):
        """Test subprotocols with mixed whitespace."""
        monkeypatch.setenv("COINBASE_WS_SUBPROTOCOLS", " v1 , v2 , v3 ")

        transport = RealTransport(config=mock_bot_config)

        with patch("websocket.create_connection") as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            transport.connect("wss://test.example.com")

            expected_subprotocols = ["v1", "v2", "v3"]
            call_args = mock_create.call_args[1]
            assert call_args["subprotocols"] == expected_subprotocols
