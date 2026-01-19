"""Tests for degradation pause monotonicity rules."""

from __future__ import annotations

from gpt_trader.features.live_trade.degradation import DegradationState


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
