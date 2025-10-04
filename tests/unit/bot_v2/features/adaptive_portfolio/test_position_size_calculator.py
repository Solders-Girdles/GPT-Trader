"""Tests for PositionSizeCalculator."""

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def tier_config() -> TierConfig:
    """Create sample tier configuration."""
    return TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),  # target_positions = 2
        min_position_size=100.0,
        strategies=["momentum"],
        risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
        trading=TradingRules(5, "cash", 2, True),
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


class TestPositionSizeCalculator:
    """Test PositionSizeCalculator."""

    def test_calculates_size_based_on_confidence(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Position size scales with confidence."""
        calculator = PositionSizeCalculator()

        # With target_positions = 2, base_size = 3000 / 2 = 1500
        # Low confidence (0.3): 1500 * 0.3 = 450
        size_low = calculator.calculate(0.3, tier_config, portfolio_snapshot)

        # Medium confidence (0.4): 1500 * 0.4 = 600
        size_medium = calculator.calculate(0.4, tier_config, portfolio_snapshot)

        assert size_medium > size_low
        assert size_low == 450.0
        assert size_medium == 600.0

    def test_enforces_minimum_position_size(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Position size respects tier minimum."""
        calculator = PositionSizeCalculator()

        # Very low confidence (0.01): 1500 * 0.01 = 15 (below min of 100)
        size = calculator.calculate(0.01, tier_config, portfolio_snapshot)

        # Should be clamped to minimum
        assert size == tier_config.min_position_size
        assert size == 100.0

    def test_enforces_maximum_position_percentage(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Position size respects maximum percentage cap."""
        calculator = PositionSizeCalculator()

        # Very high confidence (1.0): 1500 * 1.0 = 1500
        # But max is 25% of 3000 = 750
        size = calculator.calculate(1.0, tier_config, portfolio_snapshot)

        # Should be capped at 25% of portfolio
        assert size == 750.0
        assert size == portfolio_snapshot.total_value * 0.25

    def test_confidence_zero_returns_minimum_size(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Zero confidence returns minimum position size."""
        calculator = PositionSizeCalculator()

        size = calculator.calculate(0.0, tier_config, portfolio_snapshot)

        # base_size * 0.0 = 0, clamped to minimum
        assert size == tier_config.min_position_size

    def test_uses_custom_max_position_percentage(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Calculator respects custom maximum position percentage."""
        # Custom 10% max instead of default 25%
        calculator = PositionSizeCalculator(max_position_pct=0.10)

        # High confidence that would normally give large position
        size = calculator.calculate(1.0, tier_config, portfolio_snapshot)

        # Should be capped at 10% of portfolio (300) instead of 25% (750)
        assert size == 300.0
        assert size == portfolio_snapshot.total_value * 0.10

    def test_respects_tier_target_positions_for_base_size(
        self, portfolio_snapshot: PortfolioSnapshot
    ) -> None:
        """Base size calculation uses tier's target positions."""
        # Tier with target_positions = 5
        tier_with_five_targets = TierConfig(
            name="medium",
            range=(25000, 100000),
            positions=PositionConstraints(5, 15, 10, 1000.0),  # target_positions = 10
            min_position_size=500.0,
            strategies=["momentum"],
            risk=RiskProfile(1.0, 6.0, 3.0, 30.0),
            trading=TradingRules(15, "margin", 0, False),
        )

        calculator = PositionSizeCalculator()

        # base_size = 3000 / 10 = 300
        # With confidence 1.0: 300 * 1.0 = 300
        # Min is 500, so should be clamped to 500
        size = calculator.calculate(1.0, tier_with_five_targets, portfolio_snapshot)

        assert size == 500.0  # Clamped to min_position_size
