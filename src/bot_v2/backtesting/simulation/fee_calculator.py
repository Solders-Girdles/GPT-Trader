"""Fee calculation for simulated trading matching Coinbase Advanced Trade tiers."""

from decimal import Decimal

from bot_v2.backtesting.types import FEE_TIER_RATES, FeeTier


class FeeCalculator:
    """
    Calculate trading fees based on Coinbase Advanced Trade tier structure.

    Fees are calculated as a percentage of notional value (quantity * price).
    Maker fees apply when adding liquidity (limit orders).
    Taker fees apply when removing liquidity (market orders, IOC orders).

    Reference: https://help.coinbase.com/en/advanced-trade/trading-and-funding/advanced-trade-fees
    """

    def __init__(
        self,
        tier: FeeTier = FeeTier.TIER_2,
        volume_tracking: bool = False,
    ):
        """
        Initialize fee calculator.

        Args:
            tier: Initial fee tier (default: TIER_2 for $50K-$100K volume)
            volume_tracking: If True, automatically adjust tier based on 30-day volume
        """
        self.tier = tier
        self.volume_tracking = volume_tracking
        self._rolling_volume_usd: Decimal = Decimal("0")

        # Load fee rates for current tier
        self._update_rates()

    def _update_rates(self) -> None:
        """Update maker/taker rates based on current tier."""
        rates = FEE_TIER_RATES[self.tier]
        self.maker_bps = rates.maker_bps
        self.taker_bps = rates.taker_bps

    def calculate(
        self,
        notional_usd: Decimal,
        is_maker: bool,
    ) -> Decimal:
        """
        Calculate fee for a trade.

        Args:
            notional_usd: Trade notional value in USD (quantity * price)
            is_maker: True if maker order (adds liquidity), False if taker

        Returns:
            Fee amount in USD

        Examples:
            >>> calc = FeeCalculator(tier=FeeTier.TIER_2)
            >>> calc.calculate(notional_usd=Decimal("10000"), is_maker=True)
            Decimal("25.00")  # 0.25% maker fee
            >>> calc.calculate(notional_usd=Decimal("10000"), is_maker=False)
            Decimal("40.00")  # 0.40% taker fee
        """
        rate_bps = self.maker_bps if is_maker else self.taker_bps
        # Convert basis points to decimal: 25 bps = 0.0025 = 0.25%
        fee = notional_usd * rate_bps / Decimal("10000")

        # Update rolling volume if tracking
        if self.volume_tracking:
            self._add_volume(notional_usd)

        return fee

    def _add_volume(self, notional_usd: Decimal) -> None:
        """
        Add volume to rolling 30-day tracker and adjust tier if needed.

        Note: In a real implementation, this would track volume by timestamp
        and expire old trades after 30 days. For simulation purposes, we
        use a simple running total.
        """
        self._rolling_volume_usd += notional_usd

        # Check if tier should be upgraded
        old_tier = self.tier
        self.tier = self._determine_tier(self._rolling_volume_usd)

        if self.tier != old_tier:
            self._update_rates()

    def _determine_tier(self, volume_usd: Decimal) -> FeeTier:
        """Determine fee tier based on 30-day volume."""
        if volume_usd < Decimal("10000"):
            return FeeTier.TIER_0
        elif volume_usd < Decimal("50000"):
            return FeeTier.TIER_1
        elif volume_usd < Decimal("100000"):
            return FeeTier.TIER_2
        elif volume_usd < Decimal("1000000"):
            return FeeTier.TIER_3
        elif volume_usd < Decimal("15000000"):
            return FeeTier.TIER_4
        elif volume_usd < Decimal("75000000"):
            return FeeTier.TIER_5
        elif volume_usd < Decimal("250000000"):
            return FeeTier.TIER_6
        else:
            return FeeTier.TIER_7

    def get_rate_pct(self, is_maker: bool) -> Decimal:
        """
        Get current fee rate as a percentage.

        Args:
            is_maker: True for maker rate, False for taker rate

        Returns:
            Fee rate as percentage (e.g., Decimal("0.25") for 0.25%)
        """
        rate_bps = self.maker_bps if is_maker else self.taker_bps
        return rate_bps / Decimal("100")

    def reset_volume(self) -> None:
        """Reset rolling volume tracker (e.g., for monthly reset in backtesting)."""
        self._rolling_volume_usd = Decimal("0")

    @property
    def current_volume(self) -> Decimal:
        """Get current 30-day rolling volume."""
        return self._rolling_volume_usd
