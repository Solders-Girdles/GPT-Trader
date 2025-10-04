"""Symbol universe builder for strategy selector.

Provides SymbolUniverseBuilder to generate tier-appropriate symbol lists.
"""

from collections.abc import Callable

from bot_v2.features.adaptive_portfolio.types import PortfolioSnapshot, TierConfig


def _default_universe_source() -> list[str]:
    """Default symbol universe - simplified for production."""
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "NFLX",
        "SPY",
        "QQQ",
        "IWM",
        "VTI",
        "BRK-B",
        "JNJ",
        "V",
        "JPM",
        "UNH",
        "HD",
        "PG",
        "DIS",
        "MA",
        "BAC",
        "ADBE",
        "CRM",
        "PYPL",
    ]


class SymbolUniverseBuilder:
    """Builds tier-appropriate symbol universes for strategy selection."""

    def __init__(self, universe_source: Callable[[], list[str]] | None = None) -> None:
        """
        Initialize symbol universe builder.

        Args:
            universe_source: Function that returns full symbol list (default: built-in 25 symbols)
        """
        if universe_source is None:
            universe_source = _default_universe_source

        self._universe_source = universe_source

    def build_universe(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[str]:
        """
        Build symbol universe appropriate for tier.

        Args:
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state (for future enhancements)

        Returns:
            List of symbols appropriate for tier size
        """
        base_universe = self._universe_source()

        # Adjust universe size based on tier
        tier_name = tier_config.name

        if tier_name == "Micro Portfolio":
            # Small universe for micro portfolios
            return base_universe[:8]
        elif tier_name == "Small Portfolio":
            return base_universe[:12]
        elif tier_name == "Medium Portfolio":
            return base_universe[:18]
        else:  # Large Portfolio or custom tiers
            return base_universe
