"""Comprehensive tests for FeeCalculator - all fee tiers and volume tracking."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FEE_TIER_RATES, FeeTier


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
        self, tier: FeeTier, expected_maker_bps: Decimal, expected_taker_bps: Decimal
    ) -> None:
        """Verify each tier has correct maker/taker rates from FEE_TIER_RATES."""
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
            # TIER_7: 0% maker, 0.05% taker (VIP tier)
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


class TestFeeCalculatorDecimalPrecision:
    """Test decimal precision handling in fee calculations."""

    def test_fractional_notional(self) -> None:
        """Test fee calculation with fractional notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)  # 0.25% maker

        # $1234.56 * 0.0025 = $3.0864
        fee = calculator.calculate(Decimal("1234.56"), is_maker=True)
        assert fee == Decimal("3.0864")

    def test_small_notional(self) -> None:
        """Test fee calculation with small notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)

        # $10 * 0.0025 = $0.025
        fee = calculator.calculate(Decimal("10"), is_maker=True)
        assert fee == Decimal("0.025")

    def test_very_large_notional(self) -> None:
        """Test fee calculation with very large notional values."""
        calculator = FeeCalculator(tier=FeeTier.TIER_3)  # 0.15% maker

        # $10M * 0.0015 = $15,000
        fee = calculator.calculate(Decimal("10000000"), is_maker=True)
        assert fee == Decimal("15000")

    def test_zero_notional(self) -> None:
        """Test fee calculation with zero notional."""
        calculator = FeeCalculator(tier=FeeTier.TIER_2)

        fee = calculator.calculate(Decimal("0"), is_maker=True)
        assert fee == Decimal("0")


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


class TestFeeCalculatorVolumeTracking:
    """Test volume tracking and tier upgrades."""

    def test_volume_tracking_disabled_by_default(self) -> None:
        """Test that volume tracking is disabled by default."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0)

        # Trade without volume tracking - tier should not change
        calculator.calculate(Decimal("100000"), is_maker=False)
        calculator.calculate(Decimal("100000"), is_maker=False)

        assert calculator.tier == FeeTier.TIER_0
        assert calculator.current_volume == Decimal("0")

    def test_volume_tracking_enabled_accumulates(self) -> None:
        """Test volume accumulation when tracking is enabled."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("5000"), is_maker=False)
        assert calculator.current_volume == Decimal("5000")

        calculator.calculate(Decimal("3000"), is_maker=True)
        assert calculator.current_volume == Decimal("8000")

    def test_tier_upgrade_at_10k(self) -> None:
        """Test tier upgrade from TIER_0 to TIER_1 at $10K volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)
        initial_taker_bps = calculator.taker_bps

        # Below threshold - stays at TIER_0
        calculator.calculate(Decimal("9000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_0

        # Crosses $10K threshold - upgrades to TIER_1
        calculator.calculate(Decimal("2000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_1
        assert calculator.taker_bps < initial_taker_bps

    def test_tier_upgrade_at_50k(self) -> None:
        """Test tier upgrade from TIER_1 to TIER_2 at $50K volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_1, volume_tracking=True)

        # Simulate volume to cross $50K threshold
        calculator.calculate(Decimal("40000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_1

        calculator.calculate(Decimal("15000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_2

    def test_tier_upgrade_at_100k(self) -> None:
        """Test tier upgrade to TIER_3 at $100K volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("100000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_3

    def test_tier_upgrade_at_1m(self) -> None:
        """Test tier upgrade to TIER_4 at $1M volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("1000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_4

    def test_tier_upgrade_at_15m(self) -> None:
        """Test tier upgrade to TIER_5 at $15M volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("15000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_5

    def test_tier_upgrade_at_75m(self) -> None:
        """Test tier upgrade to TIER_6 at $75M volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("75000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_6

    def test_tier_upgrade_at_250m(self) -> None:
        """Test tier upgrade to TIER_7 at $250M volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("250000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_7

    def test_tier_7_maker_fee_is_zero(self) -> None:
        """Test that TIER_7 maker fee is 0 (VIP perk)."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        # Upgrade to TIER_7
        calculator.calculate(Decimal("250000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_7

        # Maker fee should be zero
        maker_fee = calculator.calculate(Decimal("1000000"), is_maker=True)
        assert maker_fee == Decimal("0")


class TestFeeCalculatorVolumeReset:
    """Test volume reset functionality."""

    def test_reset_volume_clears_accumulator(self) -> None:
        """Test that reset_volume clears the volume accumulator."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        calculator.calculate(Decimal("50000"), is_maker=False)
        assert calculator.current_volume == Decimal("50000")

        calculator.reset_volume()
        assert calculator.current_volume == Decimal("0")

    def test_reset_does_not_change_tier(self) -> None:
        """Test that reset_volume does not automatically downgrade tier."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        # Upgrade to TIER_2
        calculator.calculate(Decimal("100000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_3

        # Reset volume - tier stays (would require another determination)
        calculator.reset_volume()
        assert calculator.tier == FeeTier.TIER_3  # Tier preserved


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

        # Directly test the _determine_tier method
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

        # Multiple trades below $100K total
        for _ in range(5):
            calculator.calculate(Decimal("10000"), is_maker=True)

        assert calculator.tier == FeeTier.TIER_2
        assert calculator.current_volume == Decimal("50000")

    def test_rapid_tier_progression(self) -> None:
        """Test rapid progression through multiple tiers."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        # Single large trade that jumps multiple tiers
        calculator.calculate(Decimal("500000"), is_maker=False)

        # Should be at TIER_3 ($100K-$1M)
        assert calculator.tier == FeeTier.TIER_3

    def test_fee_rates_stay_consistent_after_update(self) -> None:
        """Test that fee rates are consistent after tier update."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0, volume_tracking=True)

        # Upgrade tier
        calculator.calculate(Decimal("100000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_3

        # Verify rates match new tier
        expected_rates = FEE_TIER_RATES[FeeTier.TIER_3]
        assert calculator.maker_bps == expected_rates.maker_bps
        assert calculator.taker_bps == expected_rates.taker_bps
