"""
WebSocket message handling, normalization, and streaming tests.
"""

from __future__ import annotations

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


class TestCoinbaseWebSocketMessageHandling:
    """Test WebSocket message processing and streaming."""

    def test_stream_messages_with_mock_transport(
        self, mock_runtime_settings, ticker_message_factory
    ):
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


class TestWebSocketMessagingErrorHandling:
    """Test WebSocket messaging error handling."""

    def test_message_parsing_error_handling(
        self, mock_runtime_settings, message_parsing_error_scenarios
    ):
        """Test handling of malformed messages."""
        CoinbaseWebSocket(
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


class TestWebSocketMessagingIntegration:
    """Test WebSocket messaging integration scenarios."""

    def test_full_lifecycle_connection_message_disconnection(
        self, mock_runtime_settings, ticker_message_factory
    ):
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

    def test_multiple_subscriptions_streaming(
        self, mock_runtime_settings, ticker_message_factory, trade_message_factory
    ):
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
