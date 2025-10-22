"""
Focused WebSocket coverage tests targeting critical functionality for 80%+ coverage.

This test suite focuses on the core WebSocket patterns that will provide the
maximum coverage improvement with minimal API complexity.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.transports import (
    MockTransport,
)
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    SequenceGuard,
    WSSubscription,
    normalize_market_message,
)


class TestWebSocketCoreCoverage:
    """Core WebSocket functionality tests for maximum coverage impact."""

    def test_websocket_initialization(self, mock_runtime_settings):
        """Test WebSocket initialization - covers __init__ method."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        assert ws.url == "wss://test.example.com"
        assert ws.connected is False
        assert ws._sub is None
        assert ws._settings is mock_runtime_settings

    def test_websocket_initialization_with_custom_params(self):
        """Test WebSocket initialization with custom parameters."""
        auth_provider = Mock()
        auth_provider.return_value = {"Authorization": "Bearer token"}

        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            max_retries=10,
            base_delay=2.0,
            liveness_timeout=60.0,
            ws_auth_provider=auth_provider,
        )

        assert ws.url == "wss://test.example.com"
        assert ws._max_retries == 10
        assert ws._base_delay == 2.0
        assert ws._liveness_timeout == 60.0
        assert ws._ws_auth_provider is auth_provider

    def test_set_transport_custom(self):
        """Test setting custom transport - covers set_transport method."""
        ws = CoinbaseWebSocket(url="wss://test.example.com")

        custom_transport = MockTransport()
        ws.set_transport(custom_transport)

        assert ws._transport is custom_transport

    def test_connect_with_mock_transport(self, mock_runtime_settings):
        """Test connection with mock transport - covers connect method start."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup mock transport that simulates successful connection
        mock_transport = Mock()
        mock_transport.connect.return_value = None
        mock_transport.connected = True
        ws.set_transport(mock_transport)

        # Test connect with headers
        headers = {"Authorization": "Bearer token"}
        ws.connect(headers=headers)

        # Should not raise exception and transport should be called
        mock_transport.connect.assert_called_once_with(headers)

    def test_connect_transport_initialization_real(self, mock_runtime_settings):
        """Test transport initialization for RealTransport - covers transport creation."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Test connect with no transport set (should initialize RealTransport)
        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport = Mock()
            mock_transport.connect.return_value = None
            mock_real.return_value = mock_transport

            ws.connect()

            mock_real.assert_called_once_with(settings=mock_runtime_settings)
            mock_transport.connect.assert_called_once()

    def test_connect_transport_initialization_noop(self, mock_runtime_settings):
        """Test NoopTransport initialization when streaming disabled."""
        # Set environment to disable streaming
        mock_runtime_settings.raw_env["DISABLE_WS_STREAMING"] = "true"

        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.NoopTransport") as mock_noop:
            mock_transport = Mock()
            mock_noop.return_value = mock_transport

            ws.connect()

            mock_noop.assert_called_once_with(settings=mock_runtime_settings)

    def test_connect_with_auth_provider(self, mock_runtime_settings):
        """Test connection with authentication provider."""
        auth_provider = Mock()
        auth_provider.return_value = {"Authorization": "Bearer token123"}

        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=auth_provider,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport = Mock()
            mock_transport.connect.return_value = None
            mock_real.return_value = mock_transport

            ws.connect()

            # Should call auth provider and pass headers to transport
            auth_provider.assert_called_once()
            expected_headers = {"Authorization": "Bearer token123"}
            mock_transport.connect.assert_called_once_with(expected_headers)

    def test_connect_error_import_error(self, mock_runtime_settings):
        """Test connection when websocket-client unavailable."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            with patch("bot_v2.features.brokerages.coinbase.ws.MockTransport") as mock_mock:
                # Simulate ImportError for websocket-client
                mock_real.side_effect = ImportError("websocket-client not available")

                mock_transport = Mock()
                mock_transport.connect.return_value = None
                mock_mock.return_value = mock_transport

                ws.connect()

                # Should fall back to MockTransport
                mock_mock.assert_called_once()
                mock_transport.connect.assert_called_once()

    def test_disconnect_basic(self, mock_runtime_settings):
        """Test basic disconnect functionality."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup transport
        mock_transport = Mock()
        ws.set_transport(mock_transport)

        ws.disconnect()

        # Should call transport disconnect and set transport to None
        mock_transport.disconnect.assert_called_once()
        assert ws._transport is None

    def test_subscribe_single_subscription(self, mock_runtime_settings):
        """Test subscribing to single channel - covers subscribe method."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup mock transport
        mock_transport = Mock()
        ws.set_transport(mock_transport)

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
        ws.subscribe(subscription)

        # Should store subscription and call transport
        assert ws._sub is subscription
        mock_transport.subscribe.assert_called_once()

    def test_subscribe_without_transport(self, mock_runtime_settings):
        """Test subscribing without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.subscribe(subscription)
        assert ws._sub is subscription

    def test_stream_messages_with_transport(self, mock_runtime_settings, ticker_message_factory):
        """Test streaming messages from transport - covers stream_messages."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup transport with test messages
        messages = [ticker_message_factory(), ticker_message_factory(symbol="ETH-USD")]
        mock_transport = MockTransport(messages=messages)
        ws.set_transport(mock_transport)

        # Stream messages
        streamed_messages = list(ws.stream_messages())

        assert len(streamed_messages) == 2
        assert streamed_messages[0]["product_id"] == "BTC-USD"
        assert streamed_messages[1]["product_id"] == "ETH-USD"

    def test_stream_messages_without_transport(self, mock_runtime_settings):
        """Test streaming messages without transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Should return empty iterator
        messages = list(ws.stream_messages())
        assert messages == []

    def test_send_message_with_transport(self, mock_runtime_settings):
        """Test sending message with active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        ws.set_transport(mock_transport)

        message = {"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}
        ws.send_message(message)

        # Should call transport send
        mock_transport.send.assert_called_once_with(json.dumps(message))

    def test_send_message_without_transport(self, mock_runtime_settings):
        """Test sending message without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        message = {"type": "ping"}

        # Should not raise exception
        ws.send_message(message)

    def test_ping_functionality(self, mock_runtime_settings):
        """Test ping functionality - covers ping method."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        ws.set_transport(mock_transport)

        ws.ping()

        # Should send ping message
        mock_transport.send.assert_called_once()
        ping_message = json.loads(mock_transport.send.call_args[0][0])
        assert ping_message["type"] == "ping"

    def test_sequence_guard_comprehensive(self):
        """Test sequence guard functionality comprehensively."""
        guard = SequenceGuard()

        # Test first message
        message1 = {"sequence": 12345, "data": "test1"}
        annotated1 = guard.annotate(message1)
        assert annotated1 == message1
        assert guard.last_seq == 12345

        # Test in-order message
        message2 = {"sequence": 12346, "data": "test2"}
        annotated2 = guard.annotate(message2)
        assert annotated2 == message2
        assert guard.last_seq == 12346

        # Test out-of-order message (should be annotated)
        message3 = {"sequence": 12344, "data": "test3"}
        annotated3 = guard.annotate(message3)
        assert annotated3["sequence"] == 12344
        assert guard.last_seq == 12344
        # Should have gap detection annotation
        assert "gap_detected" in annotated3 or "last_seq" in annotated3

        # Test reset
        guard.reset()
        assert guard.last_seq is None

    def test_message_normalization_comprehensive(self):
        """Test message normalization comprehensively."""
        # Test ticker normalization
        ticker = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "50000.00",
            "best_bid": "49900.00",
            "best_ask": "50100.00",
        }
        normalized_ticker = normalize_market_message(ticker)
        assert Decimal(normalized_ticker["price"]) == Decimal("50000.00")
        assert Decimal(normalized_ticker["best_bid"]) == Decimal("49900.00")

        # Test match normalization
        match = {
            "type": "match",
            "product_id": "BTC-USD",
            "price": "50050.00",
            "size": "0.1",
            "side": "buy",
        }
        normalized_match = normalize_market_message(match)
        assert Decimal(normalized_match["price"]) == Decimal("50050.00")
        assert Decimal(normalized_match["size"]) == Decimal("0.1")

        # Test orderbook normalization
        orderbook = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [
                ["buy", "49950.00", "0.5"],
                ["sell", "50100.00", "0.3"],
            ],
        }
        normalized_orderbook = normalize_market_message(orderbook)
        changes = normalized_orderbook["changes"]
        assert changes[0][0] == "buy"
        assert Decimal(changes[0][1]) == Decimal("49950.00")
        assert Decimal(changes[0][2]) == Decimal("0.5")

        # Test invalid message handling
        invalid = {"type": "unknown", "data": "test"}
        normalized_invalid = normalize_market_message(invalid)
        assert normalized_invalid == invalid

        # Test None handling
        result_none = normalize_market_message(None)
        assert result_none is None

    def test_ws_subscription_to_dict(self):
        """Test WSSubscription to_dict method."""
        subscription = WSSubscription(
            channels=["ticker", "matches"],
            product_ids=["BTC-USD", "ETH-USD"],
            auth_data={"token": "test_token"},
        )

        result = subscription.to_dict()

        expected = {
            "type": "subscribe",
            "channels": ["ticker", "matches"],
            "product_ids": ["BTC-USD", "ETH-USD"],
            "auth_data": {"token": "test_token"},
        }

        assert result == expected

    def test_runtime_settings_integration(self, mock_runtime_settings):
        """Test runtime settings integration throughout WebSocket lifecycle."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Test settings are used in connect
        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport = Mock()
            mock_transport.connect.return_value = None
            mock_real.return_value = mock_transport

            ws.connect()

            # Settings should be passed to transport
            mock_real.assert_called_once_with(settings=mock_runtime_settings)

    def test_error_handling_streaming(self, mock_runtime_settings):
        """Test error handling in streaming operations."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        mock_transport.stream.side_effect = ConnectionError("Connection lost")
        ws.set_transport(mock_transport)

        # Should propagate the error
        with pytest.raises(ConnectionError):
            list(ws.stream_messages())

    def test_transport_switching_scenario(self, mock_runtime_settings):
        """Test switching between transport types based on settings."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # First connect with RealTransport
        with patch("bot_v2.features.brokerages.coinbase.ws.RealTransport") as mock_real:
            mock_transport1 = Mock()
            mock_real.return_value = mock_transport1

            ws.connect()
            assert ws._transport is mock_transport1

        # Second connect with disabled streaming
        mock_runtime_settings.raw_env["DISABLE_WS_STREAMING"] = "true"

        with patch("bot_v2.features.brokerages.coinbase.ws.NoopTransport") as mock_noop:
            mock_transport2 = Mock()
            mock_noop.return_value = mock_transport2

            ws.connect()
            assert ws._transport is mock_transport2
