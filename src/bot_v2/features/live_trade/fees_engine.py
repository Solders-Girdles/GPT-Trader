"""
Fees Engine for Production Trading.

Handles fee tier awareness, maker/taker fee calculations,
and fee-adjusted PnL tracking for Coinbase derivatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FeeType(Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class FeeTier:
    """
    Coinbase fee tier information.

    Contains maker/taker rates and volume thresholds for a specific tier.
    """

    tier_name: str
    maker_rate: Decimal  # e.g., 0.005 = 0.5%
    taker_rate: Decimal  # e.g., 0.006 = 0.6%
    volume_threshold: Decimal  # 30-day volume threshold

    def get_rate(self, fee_type: FeeType) -> Decimal:
        """Get rate for specific fee type."""
        return self.maker_rate if fee_type == FeeType.MAKER else self.taker_rate


@dataclass
class FeeCalculation:
    """
    Fee calculation result for an order.
    """

    symbol: str
    notional: Decimal
    fee_type: FeeType
    fee_rate: Decimal
    fee_amount: Decimal
    tier_name: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "symbol": self.symbol,
            "notional": float(self.notional),
            "fee_type": self.fee_type.value,
            "fee_rate": float(self.fee_rate),
            "fee_amount": float(self.fee_amount),
            "tier_name": self.tier_name,
            "timestamp": self.timestamp.isoformat(),
        }


class FeeTierResolver:
    """
    Resolves current fee tier from Coinbase API.

    Manages periodic polling and tier change detection.
    """

    # Standard Coinbase Pro/Advanced Trade fee tiers
    DEFAULT_TIERS = [
        FeeTier("Tier 1", Decimal("0.006"), Decimal("0.010"), Decimal("0")),  # 0-$10k
        FeeTier("Tier 2", Decimal("0.004"), Decimal("0.006"), Decimal("10000")),  # $10k-$50k
        FeeTier("Tier 3", Decimal("0.0025"), Decimal("0.004"), Decimal("50000")),  # $50k-$100k
        FeeTier("Tier 4", Decimal("0.0015"), Decimal("0.0025"), Decimal("100000")),  # $100k-$1M
        FeeTier("Tier 5", Decimal("0.001"), Decimal("0.002"), Decimal("1000000")),  # $1M-$15M
        FeeTier("Tier 6", Decimal("0.0008"), Decimal("0.0018"), Decimal("15000000")),  # $15M-$75M
        FeeTier("Tier 7", Decimal("0.0005"), Decimal("0.0015"), Decimal("75000000")),  # $75M-$250M
        FeeTier("Tier 8", Decimal("0.0003"), Decimal("0.001"), Decimal("250000000")),  # $250M+
    ]

    def __init__(self, client: Any | None = None, refresh_interval_hours: int = 6) -> None:
        self.client = client
        self.refresh_interval = timedelta(hours=refresh_interval_hours)

        # Current tier info
        self._current_tier: FeeTier | None = None
        self._last_refresh: datetime | None = None
        self._volume_30d: Decimal | None = None

        logger.info(f"FeeTierResolver initialized - refresh every {refresh_interval_hours}h")

    async def get_current_tier(self) -> FeeTier:
        """Get current fee tier, refreshing if needed."""
        if self._should_refresh():
            await self._refresh_tier()

        return self._current_tier or self.DEFAULT_TIERS[0]

    async def get_30day_volume(self) -> Decimal | None:
        """Get 30-day trading volume."""
        if self._should_refresh():
            await self._refresh_tier()

        return self._volume_30d

    def _should_refresh(self) -> bool:
        """Check if tier data should be refreshed."""
        if not self._last_refresh:
            return True
        return (datetime.now() - self._last_refresh) >= self.refresh_interval

    async def _refresh_tier(self) -> None:
        """Refresh tier information from API."""
        try:
            if not self.client:
                logger.warning("No client available, using default tier")
                self._current_tier = self.DEFAULT_TIERS[0]
                self._last_refresh = datetime.now()
                return

            # Fetch account info (would need actual endpoint)
            # For now, simulate with default tier
            volume_30d = await self._fetch_30day_volume()
            tier = self._determine_tier_from_volume(volume_30d)

            # Check for tier changes
            old_tier_name = self._current_tier.tier_name if self._current_tier else None
            if old_tier_name and old_tier_name != tier.tier_name:
                logger.info(f"Fee tier changed: {old_tier_name} -> {tier.tier_name}")

            self._current_tier = tier
            self._volume_30d = volume_30d
            self._last_refresh = datetime.now()

            logger.debug(f"Fee tier refreshed: {tier.tier_name} (volume: ${volume_30d:,.2f})")

        except Exception as e:
            logger.error(f"Failed to refresh fee tier: {e}")
            # Fall back to default
            if not self._current_tier:
                self._current_tier = self.DEFAULT_TIERS[0]

    async def _fetch_30day_volume(self) -> Decimal:
        """Fetch 30-day volume from API."""
        # Placeholder - would integrate with actual Coinbase API
        # For now return a default volume
        return Decimal("25000")  # Puts us in Tier 2

    def _determine_tier_from_volume(self, volume: Decimal) -> FeeTier:
        """Determine fee tier based on 30-day volume."""
        for tier in reversed(self.DEFAULT_TIERS):  # Check highest tiers first
            if volume >= tier.volume_threshold:
                return tier

        return self.DEFAULT_TIERS[0]  # Default to lowest tier


class FeesEngine:
    """
    Production fees engine for trading operations.

    Calculates fees based on current tier, order type, and market conditions.
    Provides fee-aware PnL calculations and cost estimates.
    """

    def __init__(self, client: Any | None = None) -> None:
        self.tier_resolver = FeeTierResolver(client=client)
        self._fee_history: list[FeeCalculation] = []
        self._total_fees_paid: Decimal = Decimal("0")

        logger.info("FeesEngine initialized")

    async def calculate_order_fee(
        self,
        symbol: str,
        notional: Decimal,
        is_post_only: bool = False,
        is_reduce_only: bool = False,
    ) -> FeeCalculation:
        """
        Calculate expected fee for an order.

        Args:
            symbol: Trading symbol
            notional: Order notional value in USD
            is_post_only: Whether order is post-only (maker)
            is_reduce_only: Whether order reduces position

        Returns:
            Fee calculation result
        """
        current_tier = await self.tier_resolver.get_current_tier()

        # Determine fee type
        fee_type = FeeType.MAKER if is_post_only else FeeType.TAKER

        # Get rate (reduce-only orders often have lower fees)
        fee_rate = current_tier.get_rate(fee_type)
        if is_reduce_only:
            fee_rate = fee_rate * Decimal("0.8")  # 20% discount example

        # Calculate fee amount
        fee_amount = notional * fee_rate

        calculation = FeeCalculation(
            symbol=symbol,
            notional=notional,
            fee_type=fee_type,
            fee_rate=fee_rate,
            fee_amount=fee_amount,
            tier_name=current_tier.tier_name,
            timestamp=datetime.now(),
        )

        logger.debug(
            f"Fee calculated: {fee_amount:.4f} for {symbol} ({fee_type.value}, {current_tier.tier_name})"
        )

        return calculation

    async def record_actual_fee(
        self, symbol: str, notional: Decimal, actual_fee: Decimal, fee_type: FeeType
    ) -> None:
        """Record actual fee paid from trade execution."""
        current_tier = await self.tier_resolver.get_current_tier()

        calculation = FeeCalculation(
            symbol=symbol,
            notional=notional,
            fee_type=fee_type,
            fee_rate=actual_fee / notional if notional > 0 else Decimal("0"),
            fee_amount=actual_fee,
            tier_name=current_tier.tier_name,
            timestamp=datetime.now(),
        )

        self._fee_history.append(calculation)
        self._total_fees_paid += actual_fee

        # Keep last 1000 fees
        if len(self._fee_history) > 1000:
            self._fee_history = self._fee_history[-1000:]

        logger.info(f"Fee recorded: {actual_fee:.4f} for {symbol} ({fee_type.value})")

    async def get_fee_summary(self, hours_back: int = 24) -> dict[str, float | int | str | None]:
        """Get fee summary for specified period."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        recent_fees = [f for f in self._fee_history if f.timestamp >= cutoff_time]

        if not recent_fees:
            return {
                "period_hours": hours_back,
                "total_fees": 0.0,
                "maker_fees": 0.0,
                "taker_fees": 0.0,
                "trade_count": 0,
                "avg_fee_rate": 0.0,
            }

        total_fees = sum(f.fee_amount for f in recent_fees)
        maker_fees = sum(f.fee_amount for f in recent_fees if f.fee_type == FeeType.MAKER)
        taker_fees = sum(f.fee_amount for f in recent_fees if f.fee_type == FeeType.TAKER)

        total_notional = sum(f.notional for f in recent_fees)
        avg_rate = total_fees / total_notional if total_notional > 0 else Decimal("0")

        current_tier = await self.tier_resolver.get_current_tier()
        volume_30d = await self.tier_resolver.get_30day_volume()

        return {
            "period_hours": hours_back,
            "total_fees": float(total_fees),
            "maker_fees": float(maker_fees),
            "taker_fees": float(taker_fees),
            "trade_count": len(recent_fees),
            "avg_fee_rate": float(avg_rate),
            "current_tier": current_tier.tier_name,
            "maker_rate": float(current_tier.maker_rate),
            "taker_rate": float(current_tier.taker_rate),
            "volume_30d": float(volume_30d) if volume_30d else None,
            "total_lifetime_fees": float(self._total_fees_paid),
        }

    async def estimate_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        is_post_only: bool = False,
    ) -> dict[str, float | str]:
        """
        Estimate total cost of a trade including fees.

        Returns breakdown of notional, fees, and total cost.
        """
        notional = quantity * price
        fee_calculation = await self.calculate_order_fee(symbol, notional, is_post_only)

        if side.lower() == "buy":
            total_cost = notional + fee_calculation.fee_amount
        else:  # sell
            total_cost = notional - fee_calculation.fee_amount

        return {
            "symbol": symbol,
            "side": side,
            "quantity": float(quantity),
            "price": float(price),
            "notional": float(notional),
            "fee_amount": float(fee_calculation.fee_amount),
            "fee_rate": float(fee_calculation.fee_rate),
            "fee_type": fee_calculation.fee_type.value,
            "total_cost": float(total_cost),
            "tier": fee_calculation.tier_name,
        }

    def is_trade_profitable(
        self, entry_price: Decimal, exit_price: Decimal, side: str, fee_rate: Decimal
    ) -> bool:
        """
        Check if a trade would be profitable after fees.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            side: 'long' or 'short'
            fee_rate: Total fee rate (entry + exit)

        Returns:
            True if trade would be profitable
        """
        if side.lower() == "long":
            gross_return = (exit_price - entry_price) / entry_price
        else:  # short
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - fee_rate
        return net_return > 0

    async def get_minimum_profit_target(
        self,
        entry_price: Decimal,
        side: str,
        symbol: str = "BTC-USD",
        safety_margin: Decimal = Decimal("0.001"),  # 0.1% safety margin
    ) -> Decimal:
        """
        Calculate minimum profit target to cover fees.

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            symbol: Trading symbol
            safety_margin: Additional safety margin

        Returns:
            Minimum profitable exit price
        """
        current_tier = await self.tier_resolver.get_current_tier()

        # Assume worst case (taker fees for both entry and exit)
        total_fee_rate = current_tier.taker_rate * 2 + safety_margin

        if side.lower() == "long":
            min_exit_price = entry_price * (1 + total_fee_rate)
        else:  # short
            min_exit_price = entry_price * (1 - total_fee_rate)

        return min_exit_price


async def create_fees_engine(client: Any | None = None) -> FeesEngine:
    """Create and initialize fees engine."""
    engine = FeesEngine(client=client)
    await engine.tier_resolver.get_current_tier()  # Initialize tier
    logger.info("FeesEngine created and initialized")
    return engine
