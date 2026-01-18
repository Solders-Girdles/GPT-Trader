"""Tests for graceful degradation pause records and basic pause behavior."""

from __future__ import annotations

import time

from gpt_trader.features.live_trade.degradation import DegradationState, PauseRecord


class TestPauseRecord:
    """Test PauseRecord dataclass."""

    def test_pause_record_creation(self) -> None:
        record = PauseRecord(until=time.time() + 60, reason="test", allow_reduce_only=True)
        assert record.reason == "test" and record.allow_reduce_only is True


class TestDegradationStatePauseAll:
    """Test global pause functionality."""

    def test_pause_all_sets_global_pause(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test_reason")
        assert state._global_pause is not None
        assert state._global_pause.reason == "test_reason"
        assert state._global_pause.until > time.time()

    def test_is_paused_returns_true_during_global_pause(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test")
        assert state.is_paused() is True and state.is_paused(symbol="BTC-USD") is True

    def test_is_paused_returns_false_after_expiry(self) -> None:
        state = DegradationState()
        state._global_pause = PauseRecord(until=time.time() - 1, reason="expired")
        assert state.is_paused() is False

    def test_global_pause_allows_reduce_only_when_configured(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test", allow_reduce_only=True)
        assert state.is_paused(is_reduce_only=True) is False  # Allowed through
        assert state.is_paused(is_reduce_only=False) is True  # Blocked

    def test_global_pause_blocks_all_when_reduce_only_disabled(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test", allow_reduce_only=False)
        assert (
            state.is_paused(is_reduce_only=True) is True
            and state.is_paused(is_reduce_only=False) is True
        )


class TestDegradationStatePauseSymbol:
    """Test symbol-specific pause functionality."""

    def test_pause_symbol_sets_symbol_pause(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test")
        assert (
            "BTC-USD" in state._symbol_pauses and state._symbol_pauses["BTC-USD"].reason == "test"
        )

    def test_symbol_pause_only_affects_that_symbol(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test")
        assert state.is_paused(symbol="BTC-USD") is True
        assert state.is_paused(symbol="ETH-USD") is False
        assert state.is_paused() is False  # Global is not paused

    def test_symbol_pause_allows_reduce_only_when_configured(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test", allow_reduce_only=True)
        assert state.is_paused(symbol="BTC-USD", is_reduce_only=True) is False
        assert state.is_paused(symbol="BTC-USD", is_reduce_only=False) is True

    def test_symbol_pause_expires(self) -> None:
        state = DegradationState()
        state._symbol_pauses["BTC-USD"] = PauseRecord(until=time.time() - 1, reason="expired")
        assert state.is_paused(symbol="BTC-USD") is False
        assert "BTC-USD" not in state._symbol_pauses  # Cleaned up


class TestGetPauseReason:
    """Test pause reason retrieval."""

    def test_returns_global_reason_when_globally_paused(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global_test")
        assert state.get_pause_reason() == "global_test"

    def test_returns_symbol_reason_when_symbol_paused(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="symbol_test")
        assert state.get_pause_reason(symbol="BTC-USD") == "symbol_test"

    def test_returns_none_when_not_paused(self) -> None:
        state = DegradationState()
        assert state.get_pause_reason() is None and state.get_pause_reason(symbol="BTC-USD") is None
