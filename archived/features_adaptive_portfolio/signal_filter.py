"""Signal filter for strategy selector.

Provides SignalFilter to filter trading signals based on tier constraints,
portfolio state, and market constraints.
"""

from typing import Any

from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class SignalFilter:
    """Filters trading signals based on tier and market constraints."""

    def __init__(self, market_constraints: Any | None = None) -> None:
        """
        Initialize signal filter.

        Args:
            market_constraints: Market constraints object with excluded_symbols list
                               (default: None, will be provided during filter call)
        """
        self._market_constraints = market_constraints

    def filter_signals(
        self,
        signals: list[TradingSignal],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
        market_constraints: Any | None = None,
    ) -> list[TradingSignal]:
        """
        Filter signals based on tier constraints and portfolio state.

        Args:
            signals: List of trading signals to filter
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state
            market_constraints: Market constraints (overrides instance constraints)

        Returns:
            Filtered list of trading signals
        """
        # Use provided constraints or fall back to instance constraints
        constraints = market_constraints or self._market_constraints

        filtered = []

        for signal in signals:
            # Check if we already have this position
            existing_position = any(
                pos.symbol == signal.symbol for pos in portfolio_snapshot.positions
            )

            if existing_position:
                continue  # Skip if we already own it

            # Check minimum confidence for tier
            min_confidence = self._get_min_confidence_for_tier(tier_config)
            if signal.confidence < min_confidence:
                continue

            # Check if position size meets tier requirements
            if signal.target_position_size < tier_config.min_position_size:
                continue

            # Check market constraints
            if constraints and not self._meets_market_constraints(signal.symbol, constraints):
                continue

            filtered.append(signal)

        return filtered

    def _get_min_confidence_for_tier(self, tier_config: TierConfig) -> float:
        """Get minimum confidence threshold for tier."""
        # More conservative for smaller portfolios
        tier_name = tier_config.name.lower()

        if "micro" in tier_name:
            return 0.7  # High confidence required
        elif "small" in tier_name:
            return 0.6  # Moderate confidence
        elif "medium" in tier_name:
            return 0.5  # Standard confidence
        else:  # Large
            return 0.4  # More aggressive with large portfolios

    def _meets_market_constraints(self, symbol: str, constraints: Any) -> bool:
        """Check if symbol meets market constraints."""
        # Check excluded symbols
        for excluded in constraints.excluded_symbols:
            if excluded.upper() in symbol.upper():
                return False

        # Additional checks would go here (price, volume, etc.)
        # For now, assume all pass
        return True
