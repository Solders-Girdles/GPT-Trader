"""Focused NoopTransport coverage tests."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.transports import NoopTransport


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
