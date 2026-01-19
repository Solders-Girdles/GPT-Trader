"""Tests for FeeCalculator decimal precision behavior."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FeeTier


class TestFeeCalculatorDecimalPrecision:
    """Test decimal precision handling in fee calculations."""

    def test_fractional_notional(self) -> None:
        """Test fee calculation with fractional notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)  # 0.25% maker

        fee = calculator.calculate(Decimal("1234.56"), is_maker=True)
        assert fee == Decimal("3.0864")

    def test_small_notional(self) -> None:
        """Test fee calculation with small notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)

        fee = calculator.calculate(Decimal("10"), is_maker=True)
        assert fee == Decimal("0.025")

    def test_very_large_notional(self) -> None:
        """Test fee calculation with very large notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_3)  # 0.15% maker

        fee = calculator.calculate(Decimal("10000000"), is_maker=True)
        assert fee == Decimal("15000")

    def test_zero_notional(self) -> None:
        """Test fee calculation with zero notional."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)

        fee = calculator.calculate(Decimal("0"), is_maker=True)
        assert fee == Decimal("0")
