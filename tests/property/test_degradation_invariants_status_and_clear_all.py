"""Property-based tests for degradation status and clearing invariants."""

from __future__ import annotations

from hypothesis import given, seed, settings

from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.risk.config import RiskConfig
from tests.property.degradation_invariants_test_helpers import (
    pause_seconds_strategy,
    reason_strategy,
    symbol_strategy,
)


@seed(3009)
@settings(max_examples=50, deadline=None)
@given(
    symbol=symbol_strategy,
    global_reason=reason_strategy,
    symbol_reason=reason_strategy,
)
def test_clear_all_removes_all_pauses(
    symbol: str,
    global_reason: str,
    symbol_reason: str,
) -> None:
    """
    Property: clear_all should remove all pause states and counters.
    """
    state = DegradationState()
    config = RiskConfig()

    # Set up various states
    state.pause_all(seconds=60, reason=global_reason)
    state.pause_symbol(symbol=symbol, seconds=60, reason=symbol_reason)
    state.record_slippage_failure(symbol, config)
    state.record_broker_failure(config)

    # Clear all
    state.clear_all()

    # Verify everything is cleared
    assert state.is_paused() is False, "Global pause should be cleared"
    assert state.is_paused(symbol=symbol) is False, "Symbol pause should be cleared"
    assert state._broker_failures == 0, "Broker failures should be reset"
    assert len(state._slippage_failures) == 0, "Slippage failures should be reset"
    assert len(state._symbol_pauses) == 0, "Symbol pauses dict should be empty"


@seed(3010)
@settings(max_examples=100, deadline=None)
@given(
    seconds=pause_seconds_strategy,
    symbol=symbol_strategy,
)
def test_get_status_reflects_current_state(
    seconds: int,
    symbol: str,
) -> None:
    """
    Property: get_status should accurately reflect current degradation state.
    """
    state = DegradationState()

    # Initial status
    status = state.get_status()
    assert status["global_paused"] is False
    assert status["global_reason"] is None
    assert len(status["paused_symbols"]) == 0

    # After global pause
    state.pause_all(seconds=seconds, reason="test_global")
    status = state.get_status()
    assert status["global_paused"] is True
    assert status["global_reason"] == "test_global"

    # After symbol pause
    state.pause_symbol(symbol=symbol, seconds=seconds, reason="test_symbol")
    status = state.get_status()
    assert symbol in status["paused_symbols"]
    assert status["paused_symbols"][symbol]["reason"] == "test_symbol"
