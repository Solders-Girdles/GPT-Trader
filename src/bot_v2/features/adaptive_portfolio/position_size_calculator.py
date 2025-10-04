"""Position size calculator for strategy selector.

Provides PositionSizeCalculator to compute tier-appropriate position sizes
based on confidence, portfolio value, and tier constraints.
"""

from bot_v2.features.adaptive_portfolio.types import PortfolioSnapshot, TierConfig


class PositionSizeCalculator:
    """Calculates position sizes based on confidence and tier constraints."""

    def __init__(self, max_position_pct: float = 0.25) -> None:
        """
        Initialize position size calculator.

        Args:
            max_position_pct: Maximum position size as percentage of portfolio (default: 0.25 = 25%)
        """
        self._max_position_pct = max_position_pct

    def calculate(
        self, confidence: float, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> float:
        """
        Calculate position size for signal based on confidence and tier.

        Args:
            confidence: Signal confidence (0.0 to 1.0)
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            Position size in dollars
        """
        # Base position size
        target_positions = tier_config.positions.target_positions
        base_size = portfolio_snapshot.total_value / target_positions

        # Adjust for confidence
        confidence_adjusted = base_size * confidence

        # Ensure minimum size
        min_size = tier_config.min_position_size
        final_size = max(confidence_adjusted, min_size)

        # Ensure not too large
        max_size = portfolio_snapshot.total_value * self._max_position_pct
        final_size = min(final_size, max_size)

        return final_size
