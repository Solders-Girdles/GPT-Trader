"""Property-based tests for fee calculation invariants.

Tests critical fee calculation properties:
- Fees are always non-negative
- Fees scale linearly with notional
- Maker fees <= Taker fees
- Fee tiers are monotonically ordered
- Volume tracking correctly upgrades tiers
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FEE_TIER_RATES, FeeTier

# Strategies for generating valid fee parameters
notional_strategy = st.decimals(
    min_value="0", max_value="100000000", allow_nan=False, allow_infinity=False, places=2
)

positive_notional_strategy = st.decimals(
    min_value="0.01", max_value="100000000", allow_nan=False, allow_infinity=False, places=2
)

tier_strategy = st.sampled_from(list(FeeTier))


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

    assert (
        abs(fee_double - expected_double) < tolerance
    ), f"Fee not proportional: single={fee_single}, double={fee_double}, expected={expected_double}"


@seed(3003)
@settings(max_examples=100, deadline=None)
@given(
    notional=positive_notional_strategy,
    tier=tier_strategy,
)
def test_maker_fee_leq_taker_fee(
    notional: Decimal,
    tier: FeeTier,
) -> None:
    """
    Property: Maker fees should always be less than or equal to taker fees.

    This is a fundamental market structure property - makers add liquidity
    and should be rewarded with lower fees.
    """
    assume(notional > Decimal("0"))

    calculator = FeeCalculator(tier=tier, volume_tracking=False)

    maker_fee = calculator.calculate(notional, is_maker=True)
    taker_fee = calculator.calculate(notional, is_maker=False)

    assert maker_fee <= taker_fee, f"Maker fee {maker_fee} should not exceed taker fee {taker_fee}"


@seed(3004)
@settings(max_examples=100, deadline=None)
@given(
    notional=positive_notional_strategy,
)
def test_fee_tiers_monotonically_decrease(
    notional: Decimal,
) -> None:
    """
    Property: Higher volume tiers should have lower or equal fees.

    As volume increases, traders get better rates.
    """
    assume(notional > Decimal("0"))

    tiers_ordered = [
        FeeTier.TIER_0,
        FeeTier.TIER_1,
        FeeTier.TIER_2,
        FeeTier.TIER_3,
        FeeTier.TIER_4,
        FeeTier.TIER_5,
        FeeTier.TIER_6,
        FeeTier.TIER_7,
    ]

    # Calculate fees for each tier
    fees = []
    for tier in tiers_ordered:
        calculator = FeeCalculator(tier=tier, volume_tracking=False)
        fee = calculator.calculate(notional, is_maker=False)  # Use taker for comparison
        fees.append((tier, fee))

    # Each subsequent tier should have lower or equal fees
    for i in range(len(fees) - 1):
        current_tier, current_fee = fees[i]
        next_tier, next_fee = fees[i + 1]

        assert next_fee <= current_fee, (
            f"Higher tier {next_tier.value} fee {next_fee} should not exceed "
            f"lower tier {current_tier.value} fee {current_fee}"
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


@seed(3006)
@settings(max_examples=50, deadline=None)
@given(
    volumes=st.lists(
        st.decimals(
            min_value="100", max_value="1000000", allow_nan=False, allow_infinity=False, places=2
        ),
        min_size=1,
        max_size=20,
    ),
)
def test_volume_tracking_tier_upgrades(
    volumes: list[Decimal],
) -> None:
    """
    Property: Volume tracking should correctly upgrade tiers as volume accumulates.

    Tier thresholds:
    - TIER_0: < $10K
    - TIER_1: $10K-$50K
    - TIER_2: $50K-$100K
    - TIER_3: $100K-$1M
    - etc.
    """
    calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

    # Process each volume and verify tier upgrades
    cumulative_volume = Decimal("0")

    for volume in volumes:
        calculator.calculate(volume, is_maker=False)
        cumulative_volume += volume

        # Verify tier is appropriate for cumulative volume
        expected_tier = _expected_tier_for_volume(cumulative_volume)

        assert calculator.tier == expected_tier, (
            f"After volume {cumulative_volume}, tier should be {expected_tier.value}, "
            f"got {calculator.tier.value}"
        )


def _expected_tier_for_volume(volume: Decimal) -> FeeTier:
    """Determine expected tier based on volume."""
    if volume < Decimal("10000"):
        return FeeTier.TIER_0
    elif volume < Decimal("50000"):
        return FeeTier.TIER_1
    elif volume < Decimal("100000"):
        return FeeTier.TIER_2
    elif volume < Decimal("1000000"):
        return FeeTier.TIER_3
    elif volume < Decimal("15000000"):
        return FeeTier.TIER_4
    elif volume < Decimal("75000000"):
        return FeeTier.TIER_5
    elif volume < Decimal("250000000"):
        return FeeTier.TIER_6
    else:
        return FeeTier.TIER_7


@seed(3007)
@settings(max_examples=100, deadline=None)
@given(
    tier=tier_strategy,
    is_maker=st.booleans(),
)
def test_rate_percentage_conversion(
    tier: FeeTier,
    is_maker: bool,
) -> None:
    """
    Property: Rate percentage should be correctly converted from basis points.

    rate_pct = rate_bps / 100
    e.g., 25 bps = 0.25%
    """
    calculator = FeeCalculator(tier=tier, volume_tracking=False)
    rate_pct = calculator.get_rate_pct(is_maker)

    # Get rate from tier
    rates = FEE_TIER_RATES[tier]
    rate_bps = rates.maker_bps if is_maker else rates.taker_bps

    expected_pct = rate_bps / Decimal("100")

    assert rate_pct == expected_pct, (
        f"Rate PCT {rate_pct} doesn't match expected {expected_pct} " f"(rate_bps={rate_bps})"
    )


@seed(3008)
@settings(max_examples=50, deadline=None)
@given(
    initial_volume=st.decimals(
        min_value="0", max_value="10000000", allow_nan=False, allow_infinity=False, places=2
    ),
)
def test_volume_reset_clears_tracking(
    initial_volume: Decimal,
) -> None:
    """
    Property: Resetting volume should clear the tracker and potentially change tier.
    """
    calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

    # Add some volume
    calculator.calculate(initial_volume, is_maker=False)

    # Reset volume
    calculator.reset_volume()

    # Volume should be zero
    assert calculator.current_volume == Decimal(
        "0"
    ), f"After reset, volume should be 0, got {calculator.current_volume}"


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


@pytest.mark.property
class TestFeePropertyBased:
    """Grouped property-based tests for fee calculations."""

    def test_fee_tier_rates_consistency(self) -> None:
        """Verify fee tier rates are internally consistent."""
        for tier in FeeTier:
            rates = FEE_TIER_RATES[tier]

            # Maker should be <= taker
            assert (
                rates.maker_bps <= rates.taker_bps
            ), f"Tier {tier.value}: maker {rates.maker_bps} > taker {rates.taker_bps}"

            # Both should be non-negative
            assert rates.maker_bps >= Decimal("0")
            assert rates.taker_bps >= Decimal("0")

    def test_highest_tier_lowest_rates(self) -> None:
        """Verify TIER_7 has the lowest rates."""
        tier_7_rates = FEE_TIER_RATES[FeeTier.TIER_7]

        for tier in FeeTier:
            if tier != FeeTier.TIER_7:
                other_rates = FEE_TIER_RATES[tier]
                assert tier_7_rates.maker_bps <= other_rates.maker_bps
                assert tier_7_rates.taker_bps <= other_rates.taker_bps

    def test_fee_calculation_example(self) -> None:
        """Verify fee calculation matches documented example."""
        # Example from docstring: TIER_2, $10000 notional
        # Maker: 25 bps = 0.25% = $25
        # Taker: 40 bps = 0.40% = $40

        calculator = FeeCalculator(tier=FeeTier.TIER_2, volume_tracking=False)

        maker_fee = calculator.calculate(Decimal("10000"), is_maker=True)
        taker_fee = calculator.calculate(Decimal("10000"), is_maker=False)

        assert maker_fee == Decimal("25")
        assert taker_fee == Decimal("40")

    def test_volume_tracking_disabled_no_tier_change(self) -> None:
        """Verify volume tracking disabled keeps tier constant."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2, volume_tracking=False)
        initial_tier = calculator.tier

        # Process large volume
        for _ in range(100):
            calculator.calculate(Decimal("1000000"), is_maker=False)

        # Tier should not change
        assert calculator.tier == initial_tier

        # Volume should not be tracked
        assert calculator.current_volume == Decimal("0")
