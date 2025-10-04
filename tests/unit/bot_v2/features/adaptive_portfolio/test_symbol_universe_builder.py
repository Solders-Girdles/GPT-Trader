"""Tests for SymbolUniverseBuilder."""

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.symbol_universe_builder import (
    SymbolUniverseBuilder,
    _default_universe_source,
)
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Create sample portfolio snapshot."""
    return PortfolioSnapshot(
        total_value=3000.0,
        cash=2000.0,
        positions=[],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        quarterly_pnl_pct=0.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=0,
        largest_position_pct=0.0,
        sector_exposures={},
    )


class TestSymbolUniverseBuilder:
    """Test SymbolUniverseBuilder."""

    def test_micro_tier_gets_eight_symbols(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Micro tier receives 8 symbols from base universe."""
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        assert len(universe) == 8
        # Verify first 8 from default universe
        assert universe == _default_universe_source()[:8]

    def test_small_tier_gets_twelve_symbols(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Small tier receives 12 symbols from base universe."""
        tier_config = TierConfig(
            name="Small Portfolio",
            range=(5000, 25000),
            positions=PositionConstraints(3, 8, 5, 500.0),
            min_position_size=500.0,
            strategies=["momentum", "mean_reversion"],
            risk=RiskProfile(1.5, 8.0, 4.0, 40.0),
            trading=TradingRules(10, "cash", 1, True),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        assert len(universe) == 12
        # Verify first 12 from default universe
        assert universe == _default_universe_source()[:12]

    def test_medium_tier_gets_eighteen_symbols(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Medium tier receives 18 symbols from base universe."""
        tier_config = TierConfig(
            name="Medium Portfolio",
            range=(25000, 100000),
            positions=PositionConstraints(5, 15, 10, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "mean_reversion", "trend_following"],
            risk=RiskProfile(1.0, 6.0, 3.0, 30.0),
            trading=TradingRules(15, "margin", 0, False),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        assert len(universe) == 18
        # Verify first 18 from default universe
        assert universe == _default_universe_source()[:18]

    def test_large_tier_gets_all_symbols(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Large tier receives full symbol universe."""
        tier_config = TierConfig(
            name="Large Portfolio",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        # Should get all 25 symbols from default universe
        assert len(universe) == 25
        assert universe == _default_universe_source()

    def test_uses_custom_universe_source(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Builder uses custom universe source when provided."""
        custom_universe = ["CUSTOM1", "CUSTOM2", "CUSTOM3", "CUSTOM4", "CUSTOM5"]

        def custom_source() -> list[str]:
            return custom_universe

        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        builder = SymbolUniverseBuilder(universe_source=custom_source)
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        # Micro tier gets first 8 (but custom only has 5)
        assert len(universe) == 5
        assert universe == custom_universe[:8]  # Slice doesn't error if list shorter

    def test_unknown_tier_defaults_to_full_universe(
        self, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Unknown tier name defaults to full universe."""
        tier_config = TierConfig(
            name="Custom Tier Name",  # Not Micro/Small/Medium/Large
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        # Should get full universe for unknown tier
        assert len(universe) == 25
        assert universe == _default_universe_source()

    def test_default_source_when_none_provided(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Builder uses default universe source when none provided."""
        tier_config = TierConfig(
            name="Large Portfolio",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        builder = SymbolUniverseBuilder()  # No universe_source provided
        universe = builder.build_universe(tier_config, portfolio_snapshot)

        # Should use default 25-symbol universe
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "GOOGL" in universe
        assert len(universe) == 25

    def test_returns_subset_of_source_universe(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        """Smaller tiers return subset of full universe."""
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        builder = SymbolUniverseBuilder()
        universe = builder.build_universe(tier_config, portfolio_snapshot)
        full_universe = _default_universe_source()

        # All symbols in micro universe should be in full universe
        assert all(symbol in full_universe for symbol in universe)
        # Should be ordered subset (first N symbols)
        assert universe == full_universe[: len(universe)]
