"""Tests for FeeCalculator volume tracking and tier upgrades."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FeeTier


class TestFeeCalculatorVolumeTracking:
    """Test volume tracking and tier upgrades."""

    def test_volume_tracking_disabled_by_default(self) -> None:
        """Test that volume tracking is disabled by default."""
        calculator = FeeCalculator(tier=FeeTier.TIER_0)

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

        calculator.calculate(Decimal("9000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_0

        calculator.calculate(Decimal("2000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_1
        assert calculator.taker_bps < initial_taker_bps

    def test_tier_upgrade_at_50k(self) -> None:
        """Test tier upgrade from TIER_1 to TIER_2 at $50K volume."""
        calculator = FeeCalculator(tier=FeeTier.TIER_1, volume_tracking=True)

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

        calculator.calculate(Decimal("250000000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_7

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

        calculator.calculate(Decimal("100000"), is_maker=False)
        assert calculator.tier == FeeTier.TIER_3

        calculator.reset_volume()
        assert calculator.tier == FeeTier.TIER_3
