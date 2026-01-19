"""Property-based tests for fee calculation invariants."""

from __future__ import annotations

from decimal import Decimal

from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FEE_TIER_RATES, FeeTier
from tests.property.fee_invariants_test_helpers import (
    notional_strategy,
    positive_notional_strategy,
    tier_strategy,
)


@seed(3001)
@settings(max_examples=200, deadline=None)
@given(
    notional=notional_strategy,
    tier=tier_strategy,
    is_maker=st.booleans(),
)
def test_fee_non_negative(
    notional: Decimal,
    tier: FeeTier,
    is_maker: bool,
) -> None:
    """
    Property: Fees should always be non-negative.

    No matter the input, fees should never be negative.
    """
    assume(notional >= Decimal("0"))

    calculator = FeeCalculator(tier=tier, volume_tracking=False)
    fee = calculator.calculate(notional, is_maker)

    assert fee >= Decimal("0"), f"Fee {fee} should not be negative"


@seed(3002)
@settings(max_examples=200, deadline=None)
@given(
    notional=positive_notional_strategy,
    tier=tier_strategy,
    is_maker=st.booleans(),
)
def test_fee_proportional_to_notional(
    notional: Decimal,
    tier: FeeTier,
    is_maker: bool,
) -> None:
    """
    Property: Fees should be proportional to notional value.

    fee = notional * rate_bps / 10000
    Doubling notional should double the fee.
    """
    assume(notional > Decimal("0"))

    calculator = FeeCalculator(tier=tier, volume_tracking=False)

    fee_single = calculator.calculate(notional, is_maker)
    fee_double = calculator.calculate(notional * 2, is_maker)

    # Double notional should give double fee (within rounding tolerance)
    tolerance = Decimal("0.01")
    expected_double = fee_single * 2

    assert abs(fee_double - expected_double) < tolerance, (
        f"Fee not proportional: single={fee_single}, double={fee_double}, "
        f"expected={expected_double}"
    )


@seed(3005)
@settings(max_examples=100, deadline=None)
@given(
    notional=positive_notional_strategy,
    tier=tier_strategy,
    is_maker=st.booleans(),
)
def test_fee_formula_correctness(
    notional: Decimal,
    tier: FeeTier,
    is_maker: bool,
) -> None:
    """
    Property: Fee calculation should match the formula: fee = notional * rate_bps / 10000.
    """
    assume(notional > Decimal("0"))

    calculator = FeeCalculator(tier=tier, volume_tracking=False)
    actual_fee = calculator.calculate(notional, is_maker)

    # Get rate from tier
    rates = FEE_TIER_RATES[tier]
    rate_bps = rates.maker_bps if is_maker else rates.taker_bps

    # Calculate expected fee
    expected_fee = notional * rate_bps / Decimal("10000")

    assert actual_fee == expected_fee, (
        f"Fee {actual_fee} doesn't match expected {expected_fee} "
        f"(notional={notional}, rate_bps={rate_bps})"
    )


@seed(3009)
@settings(max_examples=100, deadline=None)
@given(
    notional=positive_notional_strategy,
    tier=tier_strategy,
)
def test_zero_notional_zero_fee(
    notional: Decimal,  # Not used but keeps strategy consistent
    tier: FeeTier,
) -> None:
    """
    Property: Zero notional should result in zero fee.
    """
    calculator = FeeCalculator(tier=tier, volume_tracking=False)

    maker_fee = calculator.calculate(Decimal("0"), is_maker=True)
    taker_fee = calculator.calculate(Decimal("0"), is_maker=False)

    assert maker_fee == Decimal("0")
    assert taker_fee == Decimal("0")


@seed(3010)
@settings(max_examples=50, deadline=None)
@given(
    notional1=positive_notional_strategy,
    notional2=positive_notional_strategy,
    tier=tier_strategy,
    is_maker=st.booleans(),
)
def test_fee_additivity(
    notional1: Decimal,
    notional2: Decimal,
    tier: FeeTier,
    is_maker: bool,
) -> None:
    """
    Property: Fee of sum equals sum of fees.

    fee(a + b) = fee(a) + fee(b)
    This property enables accurate aggregation of fees.
    """
    assume(notional1 > Decimal("0") and notional2 > Decimal("0"))

    calculator = FeeCalculator(tier=tier, volume_tracking=False)

    fee1 = calculator.calculate(notional1, is_maker)
    fee2 = calculator.calculate(notional2, is_maker)
    fee_combined = calculator.calculate(notional1 + notional2, is_maker)

    assert fee_combined == fee1 + fee2, (
        f"Fee additivity violated: fee({notional1} + {notional2}) = {fee_combined} "
        f"!= {fee1} + {fee2} = {fee1 + fee2}"
    )
