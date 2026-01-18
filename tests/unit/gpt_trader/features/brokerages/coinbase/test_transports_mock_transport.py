"""Focused MockTransport coverage tests."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.transports import MockTransport


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


class TestMockTransportEdgeCases:
    """MockTransport edge case coverage."""

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
