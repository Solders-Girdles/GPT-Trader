"""Base protocol for strategy handlers."""

from typing import Protocol

from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class StrategyHandler(Protocol):
    """Protocol for strategy signal generators."""

    def generate_signals(
        self,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        """
        Generate trading signals for given symbols.

        Args:
            symbols: List of symbols to analyze
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            List of trading signals
        """
        ...
