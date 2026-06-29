"""Tests for trade matching + notification/error TUI events."""

from __future__ import annotations

from textual.message import Message

from gpt_trader.tui.events import (
    ErrorOccurred,
    TradeMatcherResetRequested,
    TradeMatcherStateRequest,
)


class TestTradeMatchingEvents:
    """Test trade matching events."""

    def test_trade_matcher_reset_requested_creation(self):
        """Test TradeMatcherResetRequested event creation."""
        event = TradeMatcherResetRequested()
        assert isinstance(event, Message)

    def test_trade_matcher_state_request_creation(self):
        """Test TradeMatcherStateRequest event creation."""
        event = TradeMatcherStateRequest(request_id="test-123")
        assert isinstance(event, Message)
        assert event.request_id == "test-123"


class TestErrorNotificationEvents:
    """Test error and notification events."""

    def test_error_occurred_creation(self):
        """Test ErrorOccurred event creation."""
        exc = ValueError("Test error")
        event = ErrorOccurred(
            message="An error occurred", severity="error", context="test_module", exception=exc
        )
        assert isinstance(event, Message)
        assert event.message == "An error occurred"
        assert event.severity == "error"
        assert event.context == "test_module"
        assert event.exception == exc

    def test_error_occurred_defaults(self):
        """Test ErrorOccurred with default values."""
        event = ErrorOccurred(message="Test error")
        assert event.severity == "error"
        assert event.context == ""
        assert event.exception is None
