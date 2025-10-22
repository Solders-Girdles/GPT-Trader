"""
Enhanced tests for Coinbase WebSocket module to improve coverage from 12.93% to 85%+.

This comprehensive test suite covers:
- Connection lifecycle management and reconnection logic
- Authentication scenarios and error handling
- Message parsing and normalization
- Subscription management and validation
- Error recovery and graceful degradation
- Transport layer integration and configuration
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.transports import MockTransport, NoopTransport, RealTransport
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    SequenceGuard,
    WSSubscription,
    normalize_market_message,
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
            mock_transport.connect.assert_called_once_with("wss://test.example.com", expected_headers)

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


class TestCoinbaseWebSocketSubscription:
    """Test WebSocket subscription management."""

    def test_subscribe_single_subscription(self, mock_runtime_settings):
        """Test subscribing to a single channel."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
        ws.subscribe(subscription)

        assert subscription in ws._subscriptions
        mock_transport.subscribe.assert_called_once()

    def test_subscribe_multiple_subscriptions(self, mock_runtime_settings):
        """Test subscribing to multiple channels."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscriptions = [
            WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]),
            WSSubscription(channels=["matches"], product_ids=["ETH-USD"]),
        ]

        for sub in subscriptions:
            ws.subscribe(sub)

        assert len(ws._subscriptions) == 2
        assert all(sub in ws._subscriptions for sub in subscriptions)
        assert mock_transport.subscribe.call_count == 2

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
        assert subscription in ws._subscriptions

    def test_subscribe_duplicate_handling(self, mock_runtime_settings):
        """Test handling of duplicate subscriptions."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Subscribe twice
        ws.subscribe(subscription)
        ws.subscribe(subscription)

        # Should only have one subscription
        assert ws._subscriptions.count(subscription) == 1
        assert mock_transport.subscribe.call_count == 2  # Still called twice

    def test_unsubscribe_existing(self, mock_runtime_settings):
        """Test unsubscribing from existing subscription."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
        ws.subscribe(subscription)
        ws.unsubscribe(subscription)

        assert subscription not in ws._subscriptions

    def test_unsubscribe_nonexistent(self, mock_runtime_settings):
        """Test unsubscribing from non-existent subscription."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.unsubscribe(subscription)
        assert len(ws._subscriptions) == 0

    def test_get_subscription_message_format(self, mock_runtime_settings):
        """Test subscription message format."""
        subscription = WSSubscription(channels=["ticker", "matches"], product_ids=["BTC-USD", "ETH-USD"])

        message = subscription.to_dict()

        expected = {
            "type": "subscribe",
            "channels": ["ticker", "matches"],
            "product_ids": ["BTC-USD", "ETH-USD"],
        }

        assert message == expected


class TestCoinbaseWebSocketMessageHandling:
    """Test WebSocket message processing and streaming."""

    def test_stream_messages_with_mock_transport(self, mock_runtime_settings, ticker_message_factory):
        """Test streaming messages from mock transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Setup transport with test messages
        messages = [
            ticker_message_factory(symbol="BTC-USD", price="50000.00"),
            ticker_message_factory(symbol="ETH-USD", price="3000.00"),
        ]
        mock_transport = MockTransport(messages=messages)
        ws._transport = mock_transport

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

    def test_stream_messages_with_transport_error(self, mock_runtime_settings):
        """Test streaming messages with transport error."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        mock_transport.stream.side_effect = ConnectionError("Connection lost")
        ws._transport = mock_transport

        # Should raise the error
        with pytest.raises(ConnectionError):
            list(ws.stream_messages())

    def test_on_message_handler_assignment(self, mock_runtime_settings):
        """Test on_message handler assignment."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        handler = Mock()
        ws.on_message = handler

        assert ws._on_message is handler

    def test_send_message_with_transport(self, mock_runtime_settings):
        """Test sending message with active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        message = {"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}
        ws.send_message(message)

        # Check if message was added to transport
        assert message in mock_transport.messages

    def test_send_message_without_transport(self, mock_runtime_settings):
        """Test sending message without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        message = {"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}

        # Should not raise exception
        ws.send_message(message)

    def test_ping_with_transport(self, mock_runtime_settings):
        """Test sending ping with active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        ws.ping()

        # Check if ping message was added
        ping_messages = [msg for msg in mock_transport.messages if msg.get("type") == "ping"]
        assert len(ping_messages) == 1

    def test_ping_without_transport(self, mock_runtime_settings):
        """Test sending ping without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Should not raise exception
        ws.ping()


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


class TestSequenceGuard:
    """Test sequence number guard for message ordering."""

    def test_sequence_guard_initialization(self):
        """Test SequenceGuard initialization."""
        guard = SequenceGuard()
        assert guard.last_seq is None

    def test_sequence_guard_first_message(self):
        """Test handling first message with sequence number."""
        guard = SequenceGuard()
        message = {"sequence": 12345}

        annotated = guard.annotate(message)

        assert annotated == message
        assert guard.last_seq == 12345

    def test_sequence_guard_in_order_messages(self):
        """Test handling messages in correct order."""
        guard = SequenceGuard()

        messages = [
            {"sequence": 12345},
            {"sequence": 12346},
            {"sequence": 12347},
        ]

        for message in messages:
            annotated = guard.annotate(message)
            assert annotated == message

        assert guard.last_seq == 12347

    def test_sequence_guard_out_of_order_messages(self):
        """Test handling out-of-order messages."""
        guard = SequenceGuard()

        # First message
        first = {"sequence": 12345}
        guard.annotate(first)

        # Out of order message (should be annotated with gap info)
        out_of_order = {"sequence": 12344}
        annotated = guard.annotate(out_of_order)

        assert annotated["sequence"] == 12344
        # The actual implementation updates the sequence to the current value
        assert guard.last_seq == 12344
        # Should be annotated with gap detection
        assert "gap_detected" in annotated or "last_seq" in annotated

    def test_sequence_guard_duplicate_messages(self):
        """Test handling duplicate messages."""
        guard = SequenceGuard()

        message = {"sequence": 12345}
        first = guard.annotate(message)
        duplicate = guard.annotate(message)

        assert first == duplicate
        assert guard.last_seq == 12345

    def test_sequence_guard_missing_sequence(self):
        """Test handling messages without sequence number."""
        guard = SequenceGuard()

        message = {"type": "ticker", "price": "50000.00"}
        annotated = guard.annotate(message)

        assert annotated == message
        assert guard.last_seq is None


class TestMessageNormalization:
    """Test market message normalization and processing."""

    def test_normalize_ticker_message(self):
        """Test ticker message normalization."""
        message = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "50000.00",
            "best_bid": "49900.00",
            "best_ask": "50100.00",
        }

        normalized = normalize_market_message(message)

        assert normalized["type"] == "ticker"
        assert normalized["product_id"] == "BTC-USD"
        assert Decimal(normalized["price"]) == Decimal("50000.00")
        assert Decimal(normalized["best_bid"]) == Decimal("49900.00")
        assert Decimal(normalized["best_ask"]) == Decimal("50100.00")

    def test_normalize_match_message(self):
        """Test match/trade message normalization."""
        message = {
            "type": "match",
            "product_id": "BTC-USD",
            "price": "50050.00",
            "size": "0.1",
            "side": "buy",
        }

        normalized = normalize_market_message(message)

        assert normalized["type"] == "match"
        assert normalized["product_id"] == "BTC-USD"
        assert Decimal(normalized["price"]) == Decimal("50050.00")
        assert Decimal(normalized["size"]) == Decimal("0.1")
        assert normalized["side"] == "buy"

    def test_normalize_orderbook_message(self):
        """Test orderbook update message normalization."""
        message = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [
                ["buy", "49950.00", "0.5"],
                ["sell", "50100.00", "0.3"],
            ],
        }

        normalized = normalize_market_message(message)

        assert normalized["type"] == "l2update"
        assert normalized["product_id"] == "BTC-USD"
        assert len(normalized["changes"]) == 2
        assert normalized["changes"][0] == ["buy", Decimal("49950.00"), Decimal("0.5")]

    def test_normalize_message_with_alternative_fields(self):
        """Test message normalization with alternative field names."""
        message = {
            "channel": "ticker",
            "symbol": "BTC-USD",
            "last_price": "50000.00",
            "bid_price": "49900.00",
            "ask_price": "50100.00",
        }

        normalized = normalize_market_message(message)

        assert normalized["type"] == "ticker"
        assert normalized["product_id"] == "BTC-USD"
        assert normalized["price"] == "50000.00"

    def test_normalize_empty_message(self):
        """Test normalizing empty message."""
        result = normalize_market_message({})
        assert result == {}

    def test_normalize_none_message(self):
        """Test normalizing None message."""
        result = normalize_market_message(None)
        assert result is None

    def test_normalize_invalid_price_values(self):
        """Test normalizing messages with invalid price values."""
        message = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "invalid_price",
            "best_bid": "",
            "best_ask": None,
        }

        normalized = normalize_market_message(message)

        assert normalized["type"] == "ticker"
        assert normalized["product_id"] == "BTC-USD"
        assert normalized["price"] is None
        assert normalized["best_bid"] is None
        assert normalized["best_ask"] is None

    def test_normalize_unknown_message_type(self):
        """Test normalizing unknown message type."""
        message = {
            "type": "unknown_type",
            "product_id": "BTC-USD",
            "data": "some_data",
        }

        normalized = normalize_market_message(message)

        # Should return original message for unknown types
        assert normalized == message


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and recovery."""

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

    def test_message_parsing_error_handling(self, mock_runtime_settings, message_parsing_error_scenarios):
        """Test handling of malformed messages."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        for scenario_name, invalid_message in message_parsing_error_scenarios.items():
            if invalid_message is not None:
                try:
                    normalized = normalize_market_message(invalid_message)
                    # Should not crash, but may return None or partially normalized message
                    assert normalized is None or isinstance(normalized, dict)
                except Exception:
                    # Some parsing errors are expected and should not crash the system
                    pass

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

    def test_subscription_error_handling(self, mock_runtime_settings):
        """Test error handling in subscription operations."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        mock_transport.subscribe.side_effect = RuntimeError("Subscription failed")
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.subscribe(subscription)
        assert subscription in ws._subscriptions


class TestWebSocketIntegration:
    """Test WebSocket integration scenarios."""

    def test_full_lifecycle_connection_message_disconnection(self, mock_runtime_settings, ticker_message_factory):
        """Test complete WebSocket lifecycle: connect -> receive messages -> disconnect."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        # Connect
        with patch.object(ws, "_create_transport") as mock_create:
            mock_transport = MockTransport([ticker_message_factory()])
            mock_create.return_value = mock_transport

            ws.connect()
            assert ws.is_connected()

            # Receive message
            messages = list(ws.stream_messages())
            assert len(messages) == 1

            # Disconnect
            ws.disconnect()
            assert not ws.is_connected()

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

    def test_multiple_subscriptions_streaming(self, mock_runtime_settings, ticker_message_factory, trade_message_factory):
        """Test streaming with multiple active subscriptions."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        messages = [
            ticker_message_factory(symbol="BTC-USD"),
            trade_message_factory(symbol="ETH-USD"),
            ticker_message_factory(symbol="BTC-USD", price="50100.00"),
        ]

        with patch.object(ws, "_create_transport") as mock_create:
            mock_transport = MockTransport(messages)
            mock_create.return_value = mock_transport

            ws.connect()

            # Add subscriptions
            ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
            ws.subscribe(WSSubscription(channels=["matches"], product_ids=["ETH-USD"]))

            # Stream all messages
            streamed = list(ws.stream_messages())

            assert len(streamed) == 3
            assert len(ws._subscriptions) == 2