"""Tests for degradation state status reporting and reset behavior."""

from __future__ import annotations

from gpt_trader.features.live_trade.degradation import DegradationState


class TestClearAllAndStatus:
    """Test clear_all and get_status functionality."""

    def test_clear_all_resets_state(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global")
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="symbol")
        state._slippage_failures["BTC-USD"] = 2
        state._broker_failures = 1
        state.clear_all()
        assert state._global_pause is None
        assert len(state._symbol_pauses) == 0 and len(state._slippage_failures) == 0
        assert state._broker_failures == 0

    def test_get_status_returns_complete_state(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global_test")
        state.pause_symbol(symbol="BTC-USD", seconds=30, reason="btc_test")
        state._slippage_failures["ETH-USD"] = 2
        state._broker_failures = 1
        status = state.get_status()
        assert status["global_paused"] is True and status["global_reason"] == "global_test"
        assert status["global_remaining_seconds"] > 0 and "BTC-USD" in status["paused_symbols"]
        assert status["slippage_failures"]["ETH-USD"] == 2 and status["broker_failures"] == 1

    def test_get_status_when_not_paused(self) -> None:
        state = DegradationState()
        status = state.get_status()
        assert status["global_paused"] is False and status["global_reason"] is None
        assert status["global_remaining_seconds"] == 0 and len(status["paused_symbols"]) == 0
