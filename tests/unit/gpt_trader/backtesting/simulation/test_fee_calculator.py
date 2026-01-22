"""Tests for FeeCalculator initialization, rates, and precision."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FeeTier


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


class TestFeeCalculatorAllTiers:
    """Test fee calculation for all 8 Coinbase Advanced Trade tiers."""

    @pytest.mark.parametrize(
        "tier,expected_maker_bps,expected_taker_bps",
        [
            (FeeTier.TIER_0, Decimal("60"), Decimal("80")),  # < $10K
            (FeeTier.TIER_1, Decimal("40"), Decimal("60")),  # $10K-$50K
            (FeeTier.TIER_2, Decimal("25"), Decimal("40")),  # $50K-$100K
            (FeeTier.TIER_3, Decimal("15"), Decimal("25")),  # $100K-$1M
            (FeeTier.TIER_4, Decimal("10"), Decimal("20")),  # $1M-$15M
            (FeeTier.TIER_5, Decimal("5"), Decimal("15")),  # $15M-$75M
            (FeeTier.TIER_6, Decimal("3"), Decimal("10")),  # $75M-$250M
            (FeeTier.TIER_7, Decimal("0"), Decimal("5")),  # > $250M
        ],
    )
    def test_tier_rates_match_spec(
        self,
        tier: FeeTier,
        expected_maker_bps: Decimal,
        expected_taker_bps: Decimal,
    ) -> None:
        """Verify each tier has correct maker/taker rates."""
        calculator = FeeCalculator(tier=tier)
        assert calculator.maker_bps == expected_maker_bps
        assert calculator.taker_bps == expected_taker_bps

    @pytest.mark.parametrize(
        "tier,notional,expected_maker_fee,expected_taker_fee",
        [
            # TIER_0: 0.60% maker, 0.80% taker
            (FeeTier.TIER_0, Decimal("1000"), Decimal("6"), Decimal("8")),
            (FeeTier.TIER_0, Decimal("10000"), Decimal("60"), Decimal("80")),
            # TIER_2: 0.25% maker, 0.40% taker
            (FeeTier.TIER_2, Decimal("10000"), Decimal("25"), Decimal("40")),
            (FeeTier.TIER_2, Decimal("100000"), Decimal("250"), Decimal("400")),
            # TIER_7: 0% maker, 0.05% taker
            (FeeTier.TIER_7, Decimal("1000000"), Decimal("0"), Decimal("500")),
        ],
    )
    def test_fee_calculation_accuracy(
        self,
        tier: FeeTier,
        notional: Decimal,
        expected_maker_fee: Decimal,
        expected_taker_fee: Decimal,
    ) -> None:
        """Test fee calculation for various notional amounts."""
        calculator = FeeCalculator(tier=tier)

        maker_fee = calculator.calculate(notional, is_maker=True)
        taker_fee = calculator.calculate(notional, is_maker=False)

        assert maker_fee == expected_maker_fee
        assert taker_fee == expected_taker_fee


class TestFeeCalculatorRateConversion:
    """Test fee rate percentage conversion."""

    @pytest.mark.parametrize(
        "tier,expected_maker_pct,expected_taker_pct",
        [
            (FeeTier.TIER_0, Decimal("0.60"), Decimal("0.80")),
            (FeeTier.TIER_2, Decimal("0.25"), Decimal("0.40")),
            (FeeTier.TIER_7, Decimal("0"), Decimal("0.05")),
        ],
    )
    def test_get_rate_pct(
        self,
        tier: FeeTier,
        expected_maker_pct: Decimal,
        expected_taker_pct: Decimal,
    ) -> None:
        """Test get_rate_pct returns correct percentage values."""
        calculator = FeeCalculator(tier=tier)

        assert calculator.get_rate_pct(is_maker=True) == expected_maker_pct
        assert calculator.get_rate_pct(is_maker=False) == expected_taker_pct


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
