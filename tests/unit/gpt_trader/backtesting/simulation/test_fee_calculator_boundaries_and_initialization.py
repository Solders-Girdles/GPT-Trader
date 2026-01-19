"""Tests for FeeCalculator volume boundaries and initialization behavior."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FEE_TIER_RATES, FeeTier


class TestFeeCalculatorVolumeBoundaries:
    """Test exact volume tier boundaries."""

    @pytest.mark.parametrize(
        "volume,expected_tier",
        [
            (Decimal("0"), FeeTier.TIER_0),
            (Decimal("9999.99"), FeeTier.TIER_0),
            (Decimal("10000"), FeeTier.TIER_1),
            (Decimal("49999.99"), FeeTier.TIER_1),
            (Decimal("50000"), FeeTier.TIER_2),
            (Decimal("99999.99"), FeeTier.TIER_2),
            (Decimal("100000"), FeeTier.TIER_3),
            (Decimal("999999.99"), FeeTier.TIER_3),
            (Decimal("1000000"), FeeTier.TIER_4),
            (Decimal("14999999.99"), FeeTier.TIER_4),
            (Decimal("15000000"), FeeTier.TIER_5),
            (Decimal("74999999.99"), FeeTier.TIER_5),
            (Decimal("75000000"), FeeTier.TIER_6),
            (Decimal("249999999.99"), FeeTier.TIER_6),
            (Decimal("250000000"), FeeTier.TIER_7),
            (Decimal("1000000000"), FeeTier.TIER_7),
        ],
    )
    def test_volume_tier_boundaries(self, volume: Decimal, expected_tier: FeeTier) -> None:
        """Test tier determination at exact volume boundaries."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        actual_tier = calculator._determine_tier(volume)
        assert actual_tier == expected_tier


class TestFeeCalculatorInitialization:
    """Test FeeCalculator initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        calculator = FeeCalculator()

        assert calculator.tier == FeeTier.TIER_2  # Default tier
        assert calculator.volume_tracking is False
        assert calculator.current_volume == Decimal("0")

    def test_custom_tier_initialization(self) -> None:
        """Test initialization with custom tier."""
        calculator = FeeCalculator(tier=FeeTier.TIER_5)

        assert calculator.tier == FeeTier.TIER_5
        assert calculator.maker_bps == Decimal("5")
        assert calculator.taker_bps == Decimal("15")

    def test_volume_tracking_initialization(self) -> None:
        """Test initialization with volume tracking enabled."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        assert calculator.volume_tracking is True
        assert calculator.current_volume == Decimal("0")


class TestFeeCalculatorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_trades_same_tier(self) -> None:
        """Test multiple trades that don't trigger tier change."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2, volume_tracking=True)

        for _ in range(5):
            calculator.calculate(Decimal("10000"), is_maker=True)

        assert calculator.tier == FeeTier.TIER_2
        assert calculator.current_volume == Decimal("50000")

    def test_rapid_tier_progression(self) -> None:
        """Test rapid progression through multiple tiers."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("500000"), is_maker=False)

        assert calculator.tier == FeeTier.TIER_3

    def test_fee_rates_stay_consistent_after_update(self) -> None:
        """Test that fee rates are consistent after tier update."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("100000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_3

        expected_rates = FEE_TIER_RATES[FeeTier.TIER_3]
        assert calculator.maker_bps == expected_rates.maker_bps
        assert calculator.taker_bps == expected_rates.taker_bps
