"""Order submission rejection classification tests."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("rate_limit exceeded", "rate_limit"),
        ("HTTP 429 Too Many Requests", "rate_limit"),
        ("too many requests", "rate_limit"),
        ("Insufficient balance", "insufficient_funds"),
        ("Not enough funds", "insufficient_funds"),
        ("insufficient margin", "insufficient_funds"),
        ("Invalid size", "invalid_size"),
        ("quantity below min_size", "invalid_size"),
        ("amount too small", "invalid_size"),
        ("Invalid price", "invalid_price"),
        ("price tick increment", "invalid_price"),
        ("Request timeout", "timeout"),
        ("Connection timed out", "timeout"),
        ("deadline exceeded", "timeout"),
        ("Connection refused", "network"),
        ("Network error", "network"),
        ("socket closed", "network"),
        ("Order rejected by broker", "broker_rejected"),
        ("Request rejected", "broker_rejected"),
        ("Order failed", "unknown"),
        ("Server error", "unknown"),
        ("Something weird happened", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_rejection_reason(message: str, expected: str) -> None:
    """Test _classify_rejection_reason helper."""
    from gpt_trader.features.live_trade.execution.order_submission import (
        _classify_rejection_reason,
    )

    assert _classify_rejection_reason(message) == expected


class TestBrokerStatusClassification:
    """Tests for _classify_rejection_reason with broker status strings."""

    def test_broker_rejected_status(self) -> None:
        """Test classification of broker REJECTED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order rejected by broker: REJECTED") == "broker_status"
        assert _classify_rejection_reason("rejected by exchange") == "broker_rejected"

    def test_broker_cancelled_status(self) -> None:
        """Test classification of broker CANCELLED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        result = _classify_rejection_reason("Order rejected by broker: CANCELLED")
        assert result == "broker_status"

    def test_broker_failed_status(self) -> None:
        """Test classification of broker FAILED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order failed") == "unknown"
        assert _classify_rejection_reason("Execution failure") == "unknown"
        assert _classify_rejection_reason("FAILED status") == "unknown"

    def test_timeout_variations(self) -> None:
        """Test various timeout error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Request timeout") == "timeout"
        assert _classify_rejection_reason("Connection timed out") == "timeout"
        assert _classify_rejection_reason("deadline exceeded") == "timeout"
        assert _classify_rejection_reason("context deadline exceeded") == "timeout"

    def test_network_variations(self) -> None:
        """Test various network error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Connection refused") == "network"
        assert _classify_rejection_reason("Network error") == "network"
        assert _classify_rejection_reason("socket closed") == "network"
        assert _classify_rejection_reason("connection reset") == "network"
        assert _classify_rejection_reason("DNS resolution failed") == "network"
