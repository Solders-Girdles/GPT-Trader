"""
Market impact estimator for order execution.

Estimates market impact using a square-root impact model with adjustments
for order book depth, spread conditions, and overall liquidity state.
"""

from __future__ import annotations

from decimal import Decimal

from bot_v2.features.live_trade.liquidity_models import (
    DepthAnalysis,
    ImpactEstimate,
    LiquidityCondition,
)


class ImpactEstimator:
    """
    Estimates market impact for order execution.

    Uses square-root impact model with adjustments for:
    - Available market depth
    - Current spread conditions
    - Overall liquidity state
    """

    def __init__(self, max_impact_bps: Decimal = Decimal("50")) -> None:
        """Initialize with max acceptable impact threshold.

        Args:
            max_impact_bps: Maximum acceptable impact in basis points (default: 50 = 0.5%)
        """
        self.max_impact_bps = max_impact_bps

    def estimate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        analysis: DepthAnalysis,
        volume_metrics: dict[str, Decimal | int],
    ) -> ImpactEstimate:
        """Estimate market impact with execution recommendations.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity (base currency)
            analysis: Depth analysis from DepthAnalyzer
            volume_metrics: Volume data from MetricsTracker

        Returns:
            Impact estimate with prices, costs, and execution recommendations
        """
        # Calculate base impact using square-root model
        mid_price = (analysis.bid_price + analysis.ask_price) / 2
        notional = quantity * mid_price

        base_impact_bps = self._calculate_base_impact(notional, volume_metrics)

        # Apply depth adjustment
        impact_with_depth = self._apply_depth_adjustment(
            base_impact_bps, notional, analysis.depth_usd_5
        )

        # Apply spread and condition multipliers
        final_impact_bps = self._apply_multipliers(impact_with_depth, analysis)

        # Calculate prices and costs
        estimated_avg_price, max_impact_price = self._calculate_prices(
            side, mid_price, final_impact_bps
        )
        slippage_cost = abs(estimated_avg_price - mid_price) * quantity

        # Generate execution recommendations
        recommended_slicing, max_slice_size = self._calculate_slicing_recommendation(
            final_impact_bps, base_impact_bps, notional, mid_price
        )
        use_post_only = self._should_use_post_only(analysis.condition, final_impact_bps)

        return ImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_impact_bps=final_impact_bps,
            estimated_avg_price=estimated_avg_price,
            max_impact_price=max_impact_price,
            slippage_cost=slippage_cost,
            recommended_slicing=recommended_slicing,
            max_slice_size=max_slice_size,
            use_post_only=use_post_only,
        )

    def estimate_conservative(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
    ) -> ImpactEstimate:
        """Conservative fallback when no analysis available.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity

        Returns:
            Conservative impact estimate with defensive recommendations
        """
        return ImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_impact_bps=Decimal("100"),  # 10bps conservative
            estimated_avg_price=Decimal("0"),
            max_impact_price=Decimal("0"),
            slippage_cost=Decimal("0"),
            recommended_slicing=True,
            max_slice_size=quantity / 10,
            use_post_only=True,
        )

    def _calculate_base_impact(
        self,
        notional: Decimal,
        volume_metrics: dict[str, Decimal | int],
    ) -> Decimal:
        """Calculate base impact using square-root model.

        Args:
            notional: Order notional value (quantity * price)
            volume_metrics: Volume data including volume_15m

        Returns:
            Base impact in basis points
        """
        volume_15m = max(volume_metrics["volume_15m"], Decimal("1000"))  # Min $1k volume
        base_impact_bps = (notional / volume_15m).sqrt() * 100  # Convert to bps
        return base_impact_bps

    def _apply_depth_adjustment(
        self,
        base_impact_bps: Decimal,
        notional: Decimal,
        depth_usd_5: Decimal,
    ) -> Decimal:
        """Apply depth adjustment multiplier.

        Args:
            base_impact_bps: Base impact before adjustment
            notional: Order notional value
            depth_usd_5: Available depth within 5% of mid price

        Returns:
            Impact adjusted for depth
        """
        if notional > depth_usd_5:
            depth_multiplier = (notional / depth_usd_5).sqrt()
            return base_impact_bps * depth_multiplier
        return base_impact_bps

    def _apply_multipliers(
        self,
        impact_bps: Decimal,
        analysis: DepthAnalysis,
    ) -> Decimal:
        """Apply spread and condition multipliers.

        Args:
            impact_bps: Impact before multipliers
            analysis: Depth analysis with spread and condition

        Returns:
            Final impact after all multipliers
        """
        # Spread multiplier
        spread_multiplier = 1 + (analysis.spread_bps / 1000)

        # Condition multiplier
        condition_multiplier = {
            LiquidityCondition.EXCELLENT: Decimal("0.5"),
            LiquidityCondition.GOOD: Decimal("1.0"),
            LiquidityCondition.FAIR: Decimal("1.5"),
            LiquidityCondition.POOR: Decimal("2.0"),
            LiquidityCondition.CRITICAL: Decimal("3.0"),
        }[analysis.condition]

        return impact_bps * spread_multiplier * condition_multiplier

    def _calculate_prices(
        self,
        side: str,
        mid_price: Decimal,
        impact_bps: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Calculate estimated average price and max impact price.

        Args:
            side: 'buy' or 'sell'
            mid_price: Current mid price
            impact_bps: Final impact in basis points

        Returns:
            Tuple of (estimated_avg_price, max_impact_price)
        """
        if side == "buy":
            estimated_avg_price = mid_price * (1 + impact_bps / 10000)
            max_impact_price = mid_price * (1 + impact_bps * Decimal("1.5") / 10000)
        else:  # sell
            estimated_avg_price = mid_price * (1 - impact_bps / 10000)
            max_impact_price = mid_price * (1 - impact_bps * Decimal("1.5") / 10000)

        return estimated_avg_price, max_impact_price

    def _calculate_slicing_recommendation(
        self,
        final_impact_bps: Decimal,
        base_impact_bps: Decimal,
        notional: Decimal,
        mid_price: Decimal,
    ) -> tuple[bool, Decimal | None]:
        """Calculate slicing recommendation and max slice size.

        Args:
            final_impact_bps: Final impact after all adjustments
            base_impact_bps: Base impact before adjustments
            notional: Order notional value
            mid_price: Current mid price

        Returns:
            Tuple of (recommended_slicing, max_slice_size)
        """
        recommended_slicing = final_impact_bps > self.max_impact_bps
        max_slice_size = None

        if recommended_slicing:
            # Size slices to keep impact under threshold
            target_impact = self.max_impact_bps
            target_notional = (target_impact / base_impact_bps) ** 2 * notional
            max_slice_size = target_notional / mid_price

        return recommended_slicing, max_slice_size

    def _should_use_post_only(
        self,
        condition: LiquidityCondition,
        impact_bps: Decimal,
    ) -> bool:
        """Determine if post-only orders should be used.

        Args:
            condition: Current liquidity condition
            impact_bps: Final impact in basis points

        Returns:
            True if post-only orders recommended
        """
        return (
            condition
            in [LiquidityCondition.FAIR, LiquidityCondition.POOR, LiquidityCondition.CRITICAL]
            or impact_bps > self.max_impact_bps / 2
        )
