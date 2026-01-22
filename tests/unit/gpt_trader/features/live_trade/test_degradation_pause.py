"""Tests for graceful degradation pause records and pause behavior."""

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


class TestPauseMonotonicity:
    """Test that pauses are monotonic (only extend, never shorten)."""

    def test_global_pause_does_not_shorten_existing_pause(self) -> None:
        """A shorter pause request should not shorten an existing longer pause."""
        state = DegradationState()
        state.pause_all(seconds=300, reason="long_pause")
        original_until = state._global_pause.until

        state.pause_all(seconds=60, reason="short_pause")

        assert state._global_pause.until == original_until
        assert state._global_pause.reason == "long_pause"

    def test_global_pause_extends_existing_pause(self) -> None:
        """A longer pause request should extend an existing shorter pause."""
        state = DegradationState()
        state.pause_all(seconds=60, reason="short_pause")
        original_until = state._global_pause.until

        state.pause_all(seconds=300, reason="long_pause")

        assert state._global_pause.until > original_until
        assert state._global_pause.reason == "long_pause"

    def test_symbol_pause_does_not_shorten_existing_pause(self) -> None:
        """A shorter symbol pause should not shorten an existing longer pause."""
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="long_pause")
        original_until = state._symbol_pauses["BTC-USD"].until

        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="short_pause")

        assert state._symbol_pauses["BTC-USD"].until == original_until
        assert state._symbol_pauses["BTC-USD"].reason == "long_pause"

    def test_symbol_pause_extends_existing_pause(self) -> None:
        """A longer symbol pause request should extend an existing shorter pause."""
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="short_pause")
        original_until = state._symbol_pauses["BTC-USD"].until

        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="long_pause")

        assert state._symbol_pauses["BTC-USD"].until > original_until
        assert state._symbol_pauses["BTC-USD"].reason == "long_pause"

    def test_symbol_pause_monotonicity_independent_per_symbol(self) -> None:
        """Monotonicity is enforced independently for each symbol."""
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="btc_long")
        state.pause_symbol(symbol="ETH-USD", seconds=60, reason="eth_short")

        btc_original = state._symbol_pauses["BTC-USD"].until
        eth_original = state._symbol_pauses["ETH-USD"].until

        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="btc_shorter")
        state.pause_symbol(symbol="ETH-USD", seconds=300, reason="eth_longer")

        assert state._symbol_pauses["BTC-USD"].until == btc_original
        assert state._symbol_pauses["BTC-USD"].reason == "btc_long"
        assert state._symbol_pauses["ETH-USD"].until > eth_original
        assert state._symbol_pauses["ETH-USD"].reason == "eth_longer"
