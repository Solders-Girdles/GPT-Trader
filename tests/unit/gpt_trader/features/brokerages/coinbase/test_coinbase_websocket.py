"""
WebSocket unit tests for Coinbase integration.

Tests SequenceGuard gap detection and CoinbaseWebSocket with mock transport.
"""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket, SequenceGuard


class TestSequenceGuard:
    """Tests for SequenceGuard gap detection."""

    def test_first_message_no_gap(self):
        """First message should never have gap_detected flag."""
        guard = SequenceGuard()
        result = guard.annotate({"sequence": 1, "type": "ticker"})

        assert "gap_detected" not in result
        assert result["sequence"] == 1

    def test_consecutive_no_gap(self):
        """Consecutive sequences should not flag gap."""
        guard = SequenceGuard()

        guard.annotate({"sequence": 1})
        result = guard.annotate({"sequence": 2})

        assert "gap_detected" not in result

    def test_detects_gap(self):
        """Non-consecutive sequences should flag gap_detected."""
        guard = SequenceGuard()

        guard.annotate({"sequence": 1})
        result = guard.annotate({"sequence": 3})

        assert result.get("gap_detected") is True

    def test_large_gap(self):
        """Large sequence gaps should be detected."""
        guard = SequenceGuard()

        guard.annotate({"sequence": 100})
        result = guard.annotate({"sequence": 200})

        assert result.get("gap_detected") is True

    def test_missing_sequence_field_passes_through(self):
        """Messages without sequence field pass through unchanged."""
        guard = SequenceGuard()

        message = {"type": "heartbeat", "status": "ok"}
        result = guard.annotate(message)

        assert result == message
        assert "gap_detected" not in result

    def test_sequence_num_gap_detection(self):
        """Sequence_num gaps should be detected."""
        guard = SequenceGuard()

        guard.annotate({"sequence_num": 10})
        result = guard.annotate({"sequence_num": 12})

        assert result.get("gap_detected") is True

    def test_reset_clears_tracking(self):
        """Reset should clear sequence tracking."""
        guard = SequenceGuard()

        guard.annotate({"sequence": 100})
        guard.reset()
        result = guard.annotate({"sequence": 1})

        assert "gap_detected" not in result

    def test_preserves_message_fields(self):
        """Original message fields should be preserved."""
        guard = SequenceGuard()

        message = {"sequence": 1, "type": "ticker", "price": "50000", "symbol": "BTC-USD"}
        result = guard.annotate(message)

        assert result["type"] == "ticker"
        assert result["price"] == "50000"
        assert result["symbol"] == "BTC-USD"


class TestCoinbaseWebSocketWithMockTransport:
    """Tests for CoinbaseWebSocket using mock transport."""

    def test_initialization_defaults(self):
        """Test WebSocket initializes with correct defaults."""
        ws = CoinbaseWebSocket()

        assert ws.url == "wss://advanced-trade-ws.coinbase.com"
        assert ws.running is False
        assert ws.subscriptions == []
        assert ws._transport is None

    def test_initialization_custom_url(self):
        """Test WebSocket accepts custom URL."""
        ws = CoinbaseWebSocket(url="wss://custom.example.com")

        assert ws.url == "wss://custom.example.com"

    def test_has_sequence_guard(self):
        """WebSocket should have a SequenceGuard instance."""
        ws = CoinbaseWebSocket()

        assert hasattr(ws, "_sequence_guard")
        assert isinstance(ws._sequence_guard, SequenceGuard)

    def test_transport_none_before_connect(self):
        """Transport should be None before connect is called."""
        ws = CoinbaseWebSocket(url="wss://test.example.com")

        # Before connect, transport should be None
        assert ws._transport is None
        assert ws.running is False
