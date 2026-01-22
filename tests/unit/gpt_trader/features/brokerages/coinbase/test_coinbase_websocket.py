"""
WebSocket unit tests for Coinbase integration.

Tests SequenceGuard gap detection and CoinbaseWebSocket with mock transport.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.ws as ws_module
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


class TestCoinbaseWebSocketMessageHandling:
    """Tests for CoinbaseWebSocket message handling edge cases."""

    def test_subscribe_user_events_without_credentials_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ws = CoinbaseWebSocket(api_key=None, private_key=None)

        mock_logger = MagicMock()
        monkeypatch.setattr(ws_module, "logger", mock_logger)
        ws.subscribe_user_events()

        mock_logger.warning.assert_called_once_with(
            "Cannot subscribe to user events without API credentials"
        )
        assert ws.subscriptions == []

    def test_on_message_heartbeat_updates_timestamps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ws = CoinbaseWebSocket()

        time_calls = iter([1.0, 2.0])
        monkeypatch.setattr(ws_module.time, "time", lambda: next(time_calls))
        ws._on_message(None, json.dumps({"channel": "heartbeats"}))

        assert ws._last_message_ts == 1.0
        assert ws._last_heartbeat_ts == 2.0

    def test_on_message_sequence_gap_increments_and_passes_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[dict] = []

        def _capture(message: dict) -> None:
            captured.append(message)

        ws = CoinbaseWebSocket(on_message=_capture)

        monkeypatch.setattr(ws_module.time, "time", lambda: 1.0)
        ws._on_message(None, json.dumps({"sequence": 1, "channel": "ticker"}))
        ws._on_message(None, json.dumps({"sequence": 3, "channel": "ticker"}))

        assert ws._gap_count == 1
        assert captured[-1].get("gap_detected") is True

    def test_invalid_json_logs_error_and_preserves_last_message_ts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ws = CoinbaseWebSocket()
        ws._last_message_ts = 10.0

        mock_logger = MagicMock()
        monkeypatch.setattr(ws_module, "logger", mock_logger)
        ws._on_message(None, "not-json")

        assert ws._last_message_ts == 10.0
        mock_logger.error.assert_called_once()
