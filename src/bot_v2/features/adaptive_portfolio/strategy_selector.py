"""
Strategy selection based on portfolio tier.

Selects and configures appropriate trading strategies for each tier.
"""

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.signal_filter import SignalFilter
from bot_v2.features.adaptive_portfolio.strategy_handlers import StrategyHandler
from bot_v2.features.adaptive_portfolio.strategy_registry_factory import build_strategy_registry
from bot_v2.features.adaptive_portfolio.symbol_universe_builder import SymbolUniverseBuilder
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioConfig,
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

HAS_PANDAS = pd is not None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame as _PandasDataFrame

    DataFrame: TypeAlias = _PandasDataFrame
else:
    DataFrame: TypeAlias = Any


class StrategySelector:
    """Selects appropriate strategies based on portfolio tier."""

    def __init__(
        self,
        config: PortfolioConfig,
        data_provider: DataProvider,
        universe_builder: SymbolUniverseBuilder | None = None,
        position_size_calculator: PositionSizeCalculator | None = None,
        signal_filter: SignalFilter | None = None,
        strategy_registry: dict[str, StrategyHandler] | None = None,
    ) -> None:
        """Initialize with portfolio configuration and data provider.

        Args:
            config: Portfolio configuration with all tiers
            data_provider: Data provider for historical data
            universe_builder: Symbol universe builder (optional)
            position_size_calculator: Position size calculator (optional)
            signal_filter: Signal filter (optional)
            strategy_registry: Pre-built strategy registry (optional).
                             If None, registry will be built from tier config on-demand.
        """
        self.config = config
        self.data_provider = data_provider
        self.universe_builder = universe_builder or SymbolUniverseBuilder()
        self.position_size_calculator = position_size_calculator or PositionSizeCalculator()
        self.signal_filter = signal_filter or SignalFilter()
        self._explicit_registry = strategy_registry  # Explicit registry for tests
        self._tier_registries: dict[str, dict[str, StrategyHandler]] = {}  # Cached per-tier
        self.logger = logging.getLogger(__name__)

    def _get_strategy_registry(self, tier_config: TierConfig) -> dict[str, StrategyHandler]:
        """Get or build strategy registry for tier.

        Uses explicit registry if provided (for tests), otherwise builds
        registry from tier config and caches it.

        Args:
            tier_config: Tier configuration with strategies list

        Returns:
            Strategy registry for this tier
        """
        # Use explicit registry if provided (test injection)
        if self._explicit_registry is not None:
            return self._explicit_registry

        # Check cache
        tier_name = tier_config.name
        if tier_name in self._tier_registries:
            return self._tier_registries[tier_name]

        # Build from config and cache
        registry = build_strategy_registry(
            tier_config,
            self.data_provider,
            self.position_size_calculator,
        )
        self._tier_registries[tier_name] = registry

        self.logger.debug(
            f"Built strategy registry for tier '{tier_name}': {list(registry.keys())}"
        )

        return registry

    def generate_signals(
        self,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
        market_data: dict[str, DataFrame] | None = None,
    ) -> list[TradingSignal]:
        """
        Generate trading signals appropriate for current tier.

        Args:
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state
            market_data: Optional market data for analysis

        Returns:
            List of tier-appropriate trading signals
        """
        signals = []

        # Get symbols to analyze (simplified - would use more sophisticated universe selection)
        symbols = self.universe_builder.build_universe(tier_config, portfolio_snapshot)

        # Get strategy registry for this tier (builds from config if needed)
        strategy_registry = self._get_strategy_registry(tier_config)

        # Generate signals for each strategy in tier
        for strategy_name in tier_config.strategies:
            handler = strategy_registry.get(strategy_name)
            if handler is None:
                self.logger.warning("Unknown strategy: %s", strategy_name)
                continue

            strategy_signals = handler.generate_signals(symbols, tier_config, portfolio_snapshot)
            signals.extend(strategy_signals)

        # Filter and rank signals
        filtered_signals = self.signal_filter.filter_signals(
            signals, tier_config, portfolio_snapshot, self.config.market_constraints
        )
        ranked_signals = self._rank_signals(filtered_signals, tier_config)

        # Limit to appropriate number for tier
        max_signals = self._calculate_max_signals(tier_config, portfolio_snapshot)
        final_signals = ranked_signals[:max_signals]

        self.logger.info(
            f"Generated {len(final_signals)} signals for {tier_config.name} "
            f"using strategies: {', '.join(tier_config.strategies)}"
        )

        return final_signals

    def _rank_signals(
        self, signals: list[TradingSignal], tier_config: TierConfig
    ) -> list[TradingSignal]:
        """Rank signals by attractiveness for tier."""

        # Simple ranking by confidence for now
        # In production, would use more sophisticated ranking
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def _calculate_max_signals(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> int:
        """Calculate maximum number of new signals to generate."""

        current_positions = portfolio_snapshot.positions_count
        max_positions = tier_config.positions.max_positions
        target_positions = tier_config.positions.target_positions

        # Prioritize reaching target positions
        spots_available = max_positions - current_positions
        spots_to_target = target_positions - current_positions

        # Return the minimum of available spots and spots needed to reach target
        return max(0, min(spots_available, spots_to_target))

    def get_strategy_allocation(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> dict[str, float]:
        """
        Get recommended allocation percentages for each strategy in tier.

        Args:
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            Dictionary of strategy name -> allocation percentage
        """
        strategies = tier_config.strategies

        if len(strategies) == 1:
            return {strategies[0]: 100.0}

        # Default equal allocation
        equal_weight = 100.0 / len(strategies)
        allocation = {strategy: equal_weight for strategy in strategies}

        # Adjust allocation based on tier and market conditions
        # This is simplified - would use more sophisticated allocation in production

        if "micro" in tier_config.name.lower():
            # Conservative allocation for micro portfolios
            if "momentum" in allocation:
                allocation["momentum"] = 60.0
            if "mean_reversion" in allocation:
                allocation["mean_reversion"] = 40.0

        elif "large" in tier_config.name.lower():
            # More balanced for large portfolios
            if "ml_enhanced" in allocation:
                allocation["ml_enhanced"] = 40.0
                # Redistribute remainder equally
                remaining = 60.0 / (len(strategies) - 1)
                for strategy in strategies:
                    if strategy != "ml_enhanced":
                        allocation[strategy] = remaining

        return allocation
