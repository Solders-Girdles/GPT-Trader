"""Tier management for adaptive portfolio."""

import logging
from typing import Any

from .types import PortfolioConfig, PortfolioSnapshot, PortfolioTier, TierConfig


class TierManager:
    """Manages portfolio tier detection and transitions."""

    def __init__(self, config: PortfolioConfig) -> None:
        """Initialize with portfolio configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect_tier(self, capital: float) -> tuple[PortfolioTier, TierConfig]:
        """
        Detect appropriate tier for given capital amount.

        Args:
            capital: Current portfolio value

        Returns:
            Tuple of (PortfolioTier, TierConfig)
        """
        for tier_name, tier_config in self.config.tiers.items():
            min_val, max_val = tier_config.range
            if min_val <= capital < max_val:
                tier = PortfolioTier(tier_name)
                self.logger.info(f"Portfolio ${capital:,.0f} classified as {tier.value} tier")
                return tier, tier_config

        # Default to largest tier if above all ranges
        tier = PortfolioTier.LARGE
        tier_config = self.config.tiers["large"]
        self.logger.info(f"Portfolio ${capital:,.0f} classified as {tier.value} tier (default)")
        return tier, tier_config

    def should_transition(
        self, current_tier: PortfolioTier, current_capital: float, buffer_pct: float | None = None
    ) -> tuple[bool, PortfolioTier | None]:
        """
        Check if portfolio should transition to different tier.

        Uses hysteresis buffer to prevent frequent transitions.

        Args:
            current_tier: Current portfolio tier
            current_capital: Current portfolio value
            buffer_pct: Optional transition buffer percentage

        Returns:
            Tuple of (should_transition, target_tier)
        """
        if buffer_pct is None:
            buffer_pct = self.config.rebalancing["tier_transition_buffer_pct"]

        new_tier, _ = self.detect_tier(current_capital)

        if new_tier == current_tier:
            return False, None

        # Check if transition is significant enough
        current_config = self.config.tiers[current_tier.value]
        new_config = self.config.tiers[new_tier.value]

        # Calculate buffer zones
        if new_tier.value == "large" or current_tier.value == "large":
            # Always transition to/from large tier
            return True, new_tier

        # For other transitions, use buffer
        current_min, current_max = current_config.range
        new_min, new_max = new_config.range

        # Moving up a tier - need to be well into new range
        if current_capital > current_max:
            buffer_amount = (new_max - new_min) * buffer_pct / 100
            if current_capital >= new_min + buffer_amount:
                self.logger.info(
                    f"Tier transition UP: {current_tier.value} → {new_tier.value} "
                    f"(${current_capital:,.0f})"
                )
                return True, new_tier

        # Moving down a tier - need to be well below current range
        elif current_capital < current_min:
            current_range = current_max - current_min
            buffer_amount = current_range * buffer_pct / 100
            if current_capital <= current_min - buffer_amount:
                self.logger.info(
                    f"Tier transition DOWN: {current_tier.value} → {new_tier.value} "
                    f"(${current_capital:,.0f})"
                )
                return True, new_tier

        return False, None

    def get_tier_transitions_needed(self, portfolio_snapshot: PortfolioSnapshot) -> dict[str, Any]:
        """
        Analyze what changes are needed for tier transition.

        Args:
            portfolio_snapshot: Current portfolio state

        Returns:
            Dictionary with transition analysis
        """
        current_capital = portfolio_snapshot.total_value
        current_tier = portfolio_snapshot.current_tier

        should_transition, target_tier = self.should_transition(current_tier, current_capital)

        if not should_transition:
            return {
                "transition_needed": False,
                "current_tier": current_tier.value,
                "target_tier": None,
                "changes_needed": [],
            }

        if target_tier is None:
            return {
                "transition_needed": False,
                "current_tier": current_tier.value,
                "target_tier": None,
                "changes_needed": [],
            }

        current_config = self.config.tiers[current_tier.value]
        target_config = self.config.tiers[target_tier.value]

        changes_needed = []

        # Position count changes
        current_positions = portfolio_snapshot.positions_count
        target_positions = target_config.positions.target_positions

        if current_positions < target_positions:
            changes_needed.append(
                f"Increase positions from {current_positions} to {target_positions}"
            )
        elif current_positions > target_config.positions.max_positions:
            changes_needed.append(
                f"Reduce positions from {current_positions} to {target_config.positions.max_positions}"
            )

        # Strategy changes
        current_strategies = set(current_config.strategies)
        target_strategies = set(target_config.strategies)

        new_strategies = target_strategies - current_strategies
        removed_strategies = current_strategies - target_strategies

        if new_strategies:
            changes_needed.append(f"Add strategies: {', '.join(new_strategies)}")

        if removed_strategies:
            changes_needed.append(f"Remove strategies: {', '.join(removed_strategies)}")

        # Risk limit changes
        if current_config.risk.daily_limit_pct != target_config.risk.daily_limit_pct:
            changes_needed.append(
                f"Adjust daily risk limit from {current_config.risk.daily_limit_pct}% "
                f"to {target_config.risk.daily_limit_pct}%"
            )

        # Position sizing changes
        if current_config.min_position_size != target_config.min_position_size:
            changes_needed.append(
                f"Adjust minimum position size from ${current_config.min_position_size:,.0f} "
                f"to ${target_config.min_position_size:,.0f}"
            )

        # Trading frequency changes
        if current_config.trading.max_trades_per_week != target_config.trading.max_trades_per_week:
            changes_needed.append(
                f"Adjust max trades per week from {current_config.trading.max_trades_per_week} "
                f"to {target_config.trading.max_trades_per_week}"
            )

        return {
            "transition_needed": True,
            "current_tier": current_tier.value,
            "target_tier": target_tier.value,
            "current_capital": current_capital,
            "changes_needed": changes_needed,
            "new_config": target_config,
        }

    def validate_tier_compatibility(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> tuple[bool, list[str]]:
        """
        Validate if current portfolio is compatible with tier configuration.

        Args:
            tier_config: Target tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []

        # Check position count
        positions_count = portfolio_snapshot.positions_count
        if positions_count > tier_config.positions.max_positions:
            issues.append(
                f"Too many positions: {positions_count} > {tier_config.positions.max_positions} max"
            )

        # Check position sizes
        for position in portfolio_snapshot.positions:
            if position.position_value < tier_config.min_position_size:
                issues.append(
                    f"Position {position.symbol} too small: "
                    f"${position.position_value:,.0f} < ${tier_config.min_position_size:,.0f} minimum"
                )

        # Check risk limits
        if portfolio_snapshot.daily_pnl_pct > tier_config.risk.daily_limit_pct:
            issues.append(
                f"Daily P&L exceeds tier limit: {portfolio_snapshot.daily_pnl_pct:.2f}% > "
                f"{tier_config.risk.daily_limit_pct}% max"
            )

        # Check concentration
        if portfolio_snapshot.largest_position_pct > 25:  # General rule
            issues.append(
                f"Largest position too concentrated: {portfolio_snapshot.largest_position_pct:.1f}% > 25% recommended"
            )

        return len(issues) == 0, issues

    def recommend_tier_transition_steps(
        self, portfolio_snapshot: PortfolioSnapshot, target_tier: PortfolioTier
    ) -> list[str]:
        """
        Recommend specific steps to transition to target tier.

        Args:
            portfolio_snapshot: Current portfolio state
            target_tier: Desired tier

        Returns:
            List of recommended steps
        """
        target_config = self.config.tiers[target_tier.value]
        current_config = self.config.tiers[portfolio_snapshot.current_tier.value]

        steps = []

        # 1. Position adjustments
        current_positions = portfolio_snapshot.positions_count
        target_positions = target_config.positions.target_positions

        if current_positions > target_config.positions.max_positions:
            excess = current_positions - target_config.positions.max_positions
            steps.append(f"1. Close {excess} positions to meet tier maximum")
        elif current_positions < target_positions:
            needed = target_positions - current_positions
            steps.append(f"1. Open {needed} new positions to reach tier target")

        # 2. Position size adjustments
        small_positions = [
            pos
            for pos in portfolio_snapshot.positions
            if pos.position_value < target_config.min_position_size
        ]

        if small_positions:
            steps.append(
                f"2. Increase size of {len(small_positions)} positions to meet "
                f"${target_config.min_position_size:,.0f} minimum"
            )

        # 3. Strategy adjustments
        new_strategies = set(target_config.strategies) - set(current_config.strategies)
        if new_strategies:
            steps.append(f"3. Implement new strategies: {', '.join(new_strategies)}")

        # 4. Risk adjustments
        if target_config.risk.daily_limit_pct < current_config.risk.daily_limit_pct:
            steps.append("4. Reduce position sizes to meet lower risk limits")

        return steps
