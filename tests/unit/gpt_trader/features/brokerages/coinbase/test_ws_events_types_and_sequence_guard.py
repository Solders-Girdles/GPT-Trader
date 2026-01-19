"""Tests for EventType and Coinbase SequenceGuard."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws_events import EventType


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_values(self) -> None:
        assert EventType.TICKER.value == "ticker"
        assert EventType.LEVEL2.value == "l2_data"
        assert EventType.USER.value == "user"
        assert EventType.ERROR.value == "error"


class TestSequenceGuard:
    """Tests for SequenceGuard gap detection."""

    def test_annotate_returns_message_unchanged_without_sequence(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()
        message = {"type": "ticker", "price": "50000"}

        result = guard.annotate(message)

        assert result == message
        assert "gap_detected" not in result

    def test_annotate_first_message_no_gap(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()
        message = {"sequence": 1, "type": "ticker"}

        result = guard.annotate(message)

        assert "gap_detected" not in result

    def test_annotate_sequential_messages_no_gap(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        msg1 = guard.annotate({"sequence": 1})
        msg2 = guard.annotate({"sequence": 2})
        msg3 = guard.annotate({"sequence": 3})

        assert "gap_detected" not in msg1
        assert "gap_detected" not in msg2
        assert "gap_detected" not in msg3

    def test_annotate_gap_detected(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        guard.annotate({"sequence": 1})
        result = guard.annotate({"sequence": 5})  # Gap: 2, 3, 4 missing

        assert result.get("gap_detected") is True

    def test_reset_clears_state(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        guard.annotate({"sequence": 100})
        guard.reset()

        # After reset, first message should not detect gap
        result = guard.annotate({"sequence": 1})
        assert "gap_detected" not in result
