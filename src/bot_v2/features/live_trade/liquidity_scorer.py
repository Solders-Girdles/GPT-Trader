"""
Liquidity scoring for order book analysis.

Scores liquidity quality based on spread, depth, and imbalance metrics,
then maps composite scores to liquidity conditions.
"""

from __future__ import annotations

from decimal import Decimal

from bot_v2.features.live_trade.liquidity_models import LiquidityCondition


class LiquidityScorer:
    """
    Scores liquidity quality from order book metrics.

    Evaluates:
    - Spread tightness (0-100 score)
    - Market depth (0-100 score)
    - Order flow imbalance (0-100 score, lower imbalance = better)
    - Overall liquidity condition (EXCELLENT → CRITICAL)
    """

    def score_spread(self, spread_bps: Decimal) -> Decimal:
        """Score spread component (0-100).

        Args:
            spread_bps: Spread in basis points

        Returns:
            Score from 0-100, where 100 = tightest spread
        """
        if spread_bps <= 1:
            return Decimal("100")
        elif spread_bps <= 5:
            return Decimal("80")
        elif spread_bps <= 10:
            return Decimal("60")
        elif spread_bps <= 20:
            return Decimal("40")
        elif spread_bps <= 50:
            return Decimal("20")
        else:
            return Decimal("0")

    def score_depth(self, depth_usd: Decimal, mid_price: Decimal) -> Decimal:
        """Score depth component (0-100).

        Args:
            depth_usd: Available depth in USD
            mid_price: Current mid price (unused, kept for API compatibility)

        Returns:
            Score from 0-100, where 100 = deepest liquidity
        """
        # Score based on depth relative to typical trade sizes ($10k baseline)
        depth_score = min(depth_usd / Decimal("10000"), Decimal("1")) * Decimal("100")
        return depth_score

    def score_imbalance(self, imbalance: Decimal) -> Decimal:
        """Score imbalance component (0-100).

        Args:
            imbalance: Order flow imbalance (0-1, typically)

        Returns:
            Score from 0-100, where 100 = perfectly balanced
        """
        # Lower imbalance = better score
        return max(Decimal("0"), Decimal("100") - imbalance * Decimal("200"))

    def calculate_composite_score(
        self,
        spread_bps: Decimal,
        depth_usd_1: Decimal,
        depth_usd_5: Decimal,
        depth_imbalance: Decimal,
        mid_price: Decimal,
    ) -> Decimal:
        """Calculate composite liquidity score.

        Args:
            spread_bps: Spread in basis points
            depth_usd_1: Depth within 1% of mid price
            depth_usd_5: Depth within 5% of mid price
            depth_imbalance: Order flow imbalance
            mid_price: Current mid price

        Returns:
            Composite score (0-100)
        """
        score_components = {
            "spread": self.score_spread(spread_bps),
            "depth_1": self.score_depth(depth_usd_1, mid_price),
            "depth_5": self.score_depth(depth_usd_5, mid_price),
            "imbalance": self.score_imbalance(abs(depth_imbalance)),
        }

        composite_score = sum(score_components.values(), Decimal("0")) / Decimal(
            len(score_components)
        )
        return composite_score

    def determine_condition(self, score: Decimal) -> LiquidityCondition:
        """Determine liquidity condition from composite score.

        Args:
            score: Composite liquidity score (0-100)

        Returns:
            LiquidityCondition enum value
        """
        if score >= Decimal("80"):
            return LiquidityCondition.EXCELLENT
        elif score >= Decimal("60"):
            return LiquidityCondition.GOOD
        elif score >= Decimal("40"):
            return LiquidityCondition.FAIR
        elif score >= Decimal("20"):
            return LiquidityCondition.POOR
        else:
            return LiquidityCondition.CRITICAL
