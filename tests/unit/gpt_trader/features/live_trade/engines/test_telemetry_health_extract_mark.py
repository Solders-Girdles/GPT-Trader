"""Tests for telemetry_health.extract_mark_from_message()."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.engines.telemetry_health import extract_mark_from_message


class TestExtractMarkFromMessage:
    """Tests for extract_mark_from_message function."""

    def test_extracts_from_bid_ask(self) -> None:
        """Test extracts mark from bid/ask midpoint."""
        msg = {"best_bid": "50000", "best_ask": "50010"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("50005")

    def test_extracts_from_bid_ask_strings(self) -> None:
        """Test extracts from bid/ask as string keys."""
        msg = {"bid": "3000.50", "ask": "3001.50"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("3001")

    def test_prefers_best_bid_over_bid(self) -> None:
        """Test prefers best_bid over bid key."""
        msg = {"best_bid": "100", "bid": "90", "best_ask": "102", "ask": "92"}
        result = extract_mark_from_message(msg)

        # Should use best_bid/best_ask: (100 + 102) / 2 = 101
        assert result == Decimal("101")

    def test_extracts_from_last_price(self) -> None:
        """Test extracts from last price when no bid/ask."""
        msg = {"last": "45678.90"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("45678.90")

    def test_extracts_from_price_key(self) -> None:
        """Test extracts from price key when no bid/ask/last."""
        msg = {"price": "1234.56"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("1234.56")

    def test_returns_none_for_empty_message(self) -> None:
        """Test returns None for empty message."""
        msg: dict[str, Any] = {}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_mark(self) -> None:
        """Test returns None when mark is zero."""
        msg = {"last": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_negative_mark(self) -> None:
        """Test returns None when mark is negative."""
        msg = {"last": "-100"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_bid_ask_average(self) -> None:
        """Test returns None when bid/ask average is zero."""
        msg = {"bid": "0", "ask": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_invalid_decimal(self) -> None:
        """Test returns None for invalid decimal string."""
        msg = {"last": "not_a_number"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_handles_bid_only(self) -> None:
        """Test returns None when only bid is present."""
        msg = {"bid": "100"}
        result = extract_mark_from_message(msg)

        # Need both bid and ask for midpoint
        assert result is None

    def test_handles_ask_only(self) -> None:
        """Test returns None when only ask is present."""
        msg = {"ask": "100"}
        result = extract_mark_from_message(msg)

        # Need both bid and ask for midpoint
        assert result is None
