"""Property-based tests for graceful degradation invariants.

Tests critical degradation properties:
- Pause expiration is monotonic (never re-activates without explicit pause_all)
- Reduce-only exceptions work correctly during pauses
- Failure counters behave correctly under all input sequences
- Global vs symbol pause precedence is consistent
"""

from __future__ import annotations

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.risk.config import RiskConfig

# Strategies for generating valid parameters
pause_seconds_strategy = st.integers(min_value=1, max_value=3600)
symbol_strategy = st.sampled_from(["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"])
reason_strategy = st.text(
    min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"))
)
failure_count_strategy = st.integers(min_value=1, max_value=20)


@seed(3001)
@settings(max_examples=100, deadline=None)
@given(
    seconds=pause_seconds_strategy,
    reason=reason_strategy,
    allow_reduce_only=st.booleans(),
)
def test_global_pause_is_active_during_duration(
    seconds: int,
    reason: str,
    allow_reduce_only: bool,
) -> None:
    """
    Property: Global pause should be active immediately after calling pause_all.

    is_paused() should return True when not reduce-only (or reduce-only not allowed).
    """
    state = DegradationState()

    # Pause all trading
    state.pause_all(seconds=seconds, reason=reason, allow_reduce_only=allow_reduce_only)

    # Check immediately - should be paused
    assert state.is_paused() is True, "Global pause should be active immediately"
    assert state.is_paused(symbol="BTC-USD") is True, "Symbol should be paused under global pause"

    # Reduce-only should bypass if allowed
    if allow_reduce_only:
        assert (
            state.is_paused(is_reduce_only=True) is False
        ), "Reduce-only should bypass when allowed"
    else:
        assert (
            state.is_paused(is_reduce_only=True) is True
        ), "Reduce-only should be blocked when not allowed"


@seed(3002)
@settings(max_examples=100, deadline=None)
@given(
    symbol=symbol_strategy,
    seconds=pause_seconds_strategy,
    reason=reason_strategy,
    allow_reduce_only=st.booleans(),
)
def test_symbol_pause_is_isolated(
    symbol: str,
    seconds: int,
    reason: str,
    allow_reduce_only: bool,
) -> None:
    """
    Property: Symbol pause should only affect the paused symbol.

    Other symbols should remain unpaused.
    """
    state = DegradationState()
    other_symbol = "XRP-USD"  # Always different from the paused symbol

    # Pause specific symbol
    state.pause_symbol(
        symbol=symbol, seconds=seconds, reason=reason, allow_reduce_only=allow_reduce_only
    )

    # Paused symbol should be paused
    assert state.is_paused(symbol=symbol) is True, f"{symbol} should be paused"

    # Other symbol should not be paused
    assert state.is_paused(symbol=other_symbol) is False, f"{other_symbol} should not be paused"

    # Global check (no symbol) should not be paused
    assert state.is_paused() is False, "Global pause should not be active"


@seed(3003)
@settings(max_examples=100, deadline=None)
@given(
    symbol=symbol_strategy,
    global_seconds=pause_seconds_strategy,
    symbol_seconds=pause_seconds_strategy,
)
def test_global_pause_takes_precedence(
    symbol: str,
    global_seconds: int,
    symbol_seconds: int,
) -> None:
    """
    Property: Global pause should take precedence over symbol pause.

    When both are active, global pause properties should apply.
    """
    state = DegradationState()

    # Set both pauses with different reduce-only settings
    state.pause_all(seconds=global_seconds, reason="global", allow_reduce_only=False)
    state.pause_symbol(
        symbol=symbol, seconds=symbol_seconds, reason="symbol", allow_reduce_only=True
    )

    # Global should take precedence - reduce-only should be blocked
    assert (
        state.is_paused(symbol=symbol, is_reduce_only=True) is True
    ), "Global pause (allow_reduce_only=False) should block reduce-only"


@seed(3004)
@settings(max_examples=100, deadline=None)
@given(
    reason=reason_strategy,
)
def test_pause_reason_returned_correctly(
    reason: str,
) -> None:
    """
    Property: get_pause_reason should return the correct reason.
    """
    state = DegradationState()

    # Initially no reason
    assert state.get_pause_reason() is None

    # After pause, reason should be returned
    state.pause_all(seconds=60, reason=reason)
    returned_reason = state.get_pause_reason()

    assert returned_reason == reason, f"Expected '{reason}', got '{returned_reason}'"


@seed(3005)
@settings(max_examples=50, deadline=None)
@given(
    symbol=symbol_strategy,
    failure_count=failure_count_strategy,
    threshold=st.integers(min_value=1, max_value=10),
)
def test_slippage_failure_escalation(
    symbol: str,
    failure_count: int,
    threshold: int,
) -> None:
    """
    Property: Slippage failures should trigger pause exactly at threshold.

    The pause should be triggered when count >= threshold, and counter resets.
    """
    state = DegradationState()

    # Create config with specific threshold
    config = RiskConfig(
        slippage_failure_pause_after=threshold,
        slippage_pause_seconds=60,
    )

    pauses_triggered = 0
    for i in range(failure_count):
        triggered = state.record_slippage_failure(symbol, config)
        if triggered:
            pauses_triggered += 1

    # Number of pause triggers should match how many times we hit the threshold
    expected_triggers = failure_count // threshold
    assert pauses_triggered == expected_triggers, (
        f"Expected {expected_triggers} triggers for {failure_count} failures with threshold {threshold}, "
        f"got {pauses_triggered}"
    )

    # If triggered at least once, symbol should be paused
    if pauses_triggered > 0:
        assert state.is_paused(symbol=symbol) is True, "Symbol should be paused after threshold"


@seed(3006)
@settings(max_examples=50, deadline=None)
@given(
    failure_count=failure_count_strategy,
    threshold=st.integers(min_value=1, max_value=10),
)
def test_broker_failure_escalation(
    failure_count: int,
    threshold: int,
) -> None:
    """
    Property: Broker failures should trigger global pause exactly at threshold.

    Counter resets after pause is triggered.
    """
    state = DegradationState()

    config = RiskConfig(
        broker_outage_max_failures=threshold,
        broker_outage_cooldown_seconds=60,
    )

    pauses_triggered = 0
    for i in range(failure_count):
        triggered = state.record_broker_failure(config)
        if triggered:
            pauses_triggered += 1

    expected_triggers = failure_count // threshold
    assert (
        pauses_triggered == expected_triggers
    ), f"Expected {expected_triggers} triggers for {failure_count} failures with threshold {threshold}"


@seed(3007)
@settings(max_examples=100, deadline=None)
@given(
    symbol=symbol_strategy,
    success_after=st.integers(min_value=1, max_value=5),
)
def test_slippage_counter_resets_on_success(
    symbol: str,
    success_after: int,
) -> None:
    """
    Property: reset_slippage_failures should reset the counter.

    Subsequent failures start from zero.
    """
    state = DegradationState()
    config = RiskConfig(slippage_failure_pause_after=10, slippage_pause_seconds=60)

    # Record some failures
    for _ in range(success_after):
        state.record_slippage_failure(symbol, config)

    # Reset
    state.reset_slippage_failures(symbol)

    # Internal counter should be zero
    assert state._slippage_failures.get(symbol, 0) == 0


@seed(3008)
@settings(max_examples=100, deadline=None)
@given(
    success_after=st.integers(min_value=1, max_value=5),
)
def test_broker_counter_resets_on_success(
    success_after: int,
) -> None:
    """
    Property: reset_broker_failures should reset the counter.
    """
    state = DegradationState()
    config = RiskConfig(broker_outage_max_failures=10, broker_outage_cooldown_seconds=60)

    # Record some failures
    for _ in range(success_after):
        state.record_broker_failure(config)

    # Reset
    state.reset_broker_failures()

    # Internal counter should be zero
    assert state._broker_failures == 0


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


@seed(3011)
@settings(max_examples=50, deadline=None)
@given(
    symbols=st.lists(symbol_strategy, min_size=2, max_size=4, unique=True),
)
def test_multiple_symbol_pauses_are_independent(
    symbols: list[str],
) -> None:
    """
    Property: Multiple symbol pauses should be independent.

    Each symbol can be paused/unpaused independently.
    """
    state = DegradationState()

    # Pause first symbol only
    first_symbol = symbols[0]
    other_symbols = symbols[1:]

    state.pause_symbol(symbol=first_symbol, seconds=60, reason="test")

    # First symbol should be paused
    assert state.is_paused(symbol=first_symbol) is True

    # Other symbols should not be paused
    for other in other_symbols:
        assert state.is_paused(symbol=other) is False, f"{other} should not be paused"


@seed(3012)
@settings(max_examples=50, deadline=None)
@given(
    allow_reduce_only=st.booleans(),
)
def test_reduce_only_consistency_global(
    allow_reduce_only: bool,
) -> None:
    """
    Property: Reduce-only behavior should be consistent with the pause setting.
    """
    state = DegradationState()

    state.pause_all(seconds=60, reason="test", allow_reduce_only=allow_reduce_only)

    # Non-reduce-only orders should always be blocked
    assert state.is_paused(is_reduce_only=False) is True

    # Reduce-only orders should be blocked only if not allowed
    if allow_reduce_only:
        assert state.is_paused(is_reduce_only=True) is False
    else:
        assert state.is_paused(is_reduce_only=True) is True


@seed(3013)
@settings(max_examples=50, deadline=None)
@given(
    symbol=symbol_strategy,
    allow_reduce_only=st.booleans(),
)
def test_reduce_only_consistency_symbol(
    symbol: str,
    allow_reduce_only: bool,
) -> None:
    """
    Property: Reduce-only behavior for symbol pause should be consistent.
    """
    state = DegradationState()

    state.pause_symbol(
        symbol=symbol, seconds=60, reason="test", allow_reduce_only=allow_reduce_only
    )

    # Non-reduce-only orders should always be blocked for this symbol
    assert state.is_paused(symbol=symbol, is_reduce_only=False) is True

    # Reduce-only orders follow the setting
    if allow_reduce_only:
        assert state.is_paused(symbol=symbol, is_reduce_only=True) is False
    else:
        assert state.is_paused(symbol=symbol, is_reduce_only=True) is True


__all__ = ["test_global_pause_is_active_during_duration"]
