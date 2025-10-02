"""
Comprehensive tests for TierManager.

Tests cover:
- Tier detection based on capital
- Tier transition logic with hysteresis buffer
- Transition analysis and change identification
- Tier compatibility validation
- Transition step recommendations
- Edge cases (boundary conditions, buffers)
"""

import pytest

from bot_v2.features.adaptive_portfolio.tier_manager import TierManager
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioConfig,
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
    CostStructure,
    MarketConstraints,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def portfolio_config():
    """Create test portfolio configuration with multiple tiers."""
    micro_tier = TierConfig(
        name="micro",
        range=(0, 2000),
        positions=PositionConstraints(
            min_positions=1,
            max_positions=5,
            target_positions=3,
            min_position_size=100,
        ),
        min_position_size=100,
        strategies=["momentum"],
        risk=RiskProfile(
            daily_limit_pct=2.0,
            quarterly_limit_pct=10.0,
            position_stop_loss_pct=5.0,
            max_sector_exposure_pct=40.0,
        ),
        trading=TradingRules(
            max_trades_per_week=5,
            account_type="cash",
            settlement_days=2,
            pdt_compliant=True,
        ),
    )

    small_tier = TierConfig(
        name="small",
        range=(2000, 10000),
        positions=PositionConstraints(
            min_positions=3,
            max_positions=10,
            target_positions=6,
            min_position_size=200,
        ),
        min_position_size=200,
        strategies=["momentum", "mean_reversion"],
        risk=RiskProfile(
            daily_limit_pct=3.0,
            quarterly_limit_pct=15.0,
            position_stop_loss_pct=7.0,
            max_sector_exposure_pct=50.0,
        ),
        trading=TradingRules(
            max_trades_per_week=15,
            account_type="margin",
            settlement_days=0,
            pdt_compliant=False,
        ),
    )

    medium_tier = TierConfig(
        name="medium",
        range=(10000, 50000),
        positions=PositionConstraints(
            min_positions=5,
            max_positions=20,
            target_positions=10,
            min_position_size=500,
        ),
        min_position_size=500,
        strategies=["momentum", "mean_reversion", "breakout"],
        risk=RiskProfile(
            daily_limit_pct=4.0,
            quarterly_limit_pct=20.0,
            position_stop_loss_pct=10.0,
            max_sector_exposure_pct=60.0,
        ),
        trading=TradingRules(
            max_trades_per_week=30,
            account_type="margin",
            settlement_days=0,
            pdt_compliant=False,
        ),
    )

    large_tier = TierConfig(
        name="large",
        range=(50000, float('inf')),
        positions=PositionConstraints(
            min_positions=10,
            max_positions=50,
            target_positions=20,
            min_position_size=1000,
        ),
        min_position_size=1000,
        strategies=["momentum", "mean_reversion", "breakout", "statistical_arbitrage"],
        risk=RiskProfile(
            daily_limit_pct=5.0,
            quarterly_limit_pct=25.0,
            position_stop_loss_pct=12.0,
            max_sector_exposure_pct=70.0,
        ),
        trading=TradingRules(
            max_trades_per_week=100,
            account_type="margin",
            settlement_days=0,
            pdt_compliant=False,
        ),
    )

    return PortfolioConfig(
        version="1.0",
        last_updated="2025-01-01",
        description="Test config",
        tiers={
            "micro": micro_tier,
            "small": small_tier,
            "medium": medium_tier,
            "large": large_tier,
        },
        costs=CostStructure(
            commission_per_trade=0.0,
            spread_estimate_pct=0.05,
            slippage_pct=0.1,
            financing_rate_annual_pct=5.0,
        ),
        market_constraints=MarketConstraints(
            min_share_price=1.0,
            max_share_price=1000.0,
            min_daily_volume=100000,
            excluded_sectors=[],
            excluded_symbols=[],
            market_hours_only=True,
        ),
        validation={"max_position_size_pct": 25.0},
        rebalancing={"tier_transition_buffer_pct": 10.0},
    )


@pytest.fixture
def tier_manager(portfolio_config):
    """Create TierManager instance."""
    return TierManager(portfolio_config)


# ============================================================================
# Test: Tier Detection
# ============================================================================


class TestTierDetection:
    """Test detect_tier method."""

    def test_detects_micro_tier(self, tier_manager):
        """Test detection of micro tier."""
        tier, config = tier_manager.detect_tier(capital=1000.0)

        assert tier == PortfolioTier.MICRO
        assert config.name == "micro"

    def test_detects_small_tier(self, tier_manager):
        """Test detection of small tier."""
        tier, config = tier_manager.detect_tier(capital=5000.0)

        assert tier == PortfolioTier.SMALL
        assert config.name == "small"

    def test_detects_medium_tier(self, tier_manager):
        """Test detection of medium tier."""
        tier, config = tier_manager.detect_tier(capital=25000.0)

        assert tier == PortfolioTier.MEDIUM
        assert config.name == "medium"

    def test_detects_large_tier(self, tier_manager):
        """Test detection of large tier."""
        tier, config = tier_manager.detect_tier(capital=100000.0)

        assert tier == PortfolioTier.LARGE
        assert config.name == "large"

    def test_tier_boundaries_lower(self, tier_manager):
        """Test tier detection at lower boundaries."""
        # At exact boundary (2000) should be small tier
        tier, _ = tier_manager.detect_tier(capital=2000.0)
        assert tier == PortfolioTier.SMALL

    def test_tier_boundaries_upper(self, tier_manager):
        """Test tier detection at upper boundaries."""
        # Just below boundary (1999) should be micro tier
        tier, _ = tier_manager.detect_tier(capital=1999.0)
        assert tier == PortfolioTier.MICRO

    def test_very_large_capital_defaults_to_large(self, tier_manager):
        """Test that very large capital defaults to large tier."""
        tier, _ = tier_manager.detect_tier(capital=1000000.0)
        assert tier == PortfolioTier.LARGE


# ============================================================================
# Test: Tier Transitions
# ============================================================================


class TestTierTransitions:
    """Test should_transition method."""

    def test_no_transition_within_same_tier(self, tier_manager):
        """Test no transition when capital stays within tier."""
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.MICRO,
            current_capital=1500.0  # Within micro range
        )

        assert should_transition is False
        assert target is None

    def test_transition_up_with_buffer(self, tier_manager):
        """Test upward transition requires buffer."""
        # Small tier starts at 2000, with 10% buffer = 2800
        # Should not transition at 2100
        should_transition, _ = tier_manager.should_transition(
            current_tier=PortfolioTier.MICRO,
            current_capital=2100.0
        )
        assert should_transition is False

        # Should transition at 2900 (above buffer)
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.MICRO,
            current_capital=2900.0
        )
        assert should_transition is True
        assert target == PortfolioTier.SMALL

    def test_transition_down_with_buffer(self, tier_manager):
        """Test downward transition requires buffer."""
        # Small tier starts at 2000, with 10% buffer = 200
        # Should not transition at 1900
        should_transition, _ = tier_manager.should_transition(
            current_tier=PortfolioTier.SMALL,
            current_capital=1900.0
        )
        assert should_transition is False

        # Should transition at 1100 (below buffer threshold of 1200)
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.SMALL,
            current_capital=1100.0
        )
        assert should_transition is True
        assert target == PortfolioTier.MICRO

    def test_transition_to_large_tier_immediate(self, tier_manager):
        """Test transition to large tier happens immediately."""
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.MEDIUM,
            current_capital=50100.0  # Just over large tier boundary
        )

        # Should transition immediately to large tier
        assert should_transition is True
        assert target == PortfolioTier.LARGE

    def test_transition_from_large_tier_immediate(self, tier_manager):
        """Test transition from large tier happens immediately."""
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.LARGE,
            current_capital=49000.0  # Below large tier minimum
        )

        # Should transition immediately from large tier
        assert should_transition is True
        assert target == PortfolioTier.MEDIUM

    def test_custom_buffer_percentage(self, tier_manager):
        """Test using custom buffer percentage."""
        # Use 5% buffer instead of default 10%
        should_transition, target = tier_manager.should_transition(
            current_tier=PortfolioTier.MICRO,
            current_capital=2500.0,  # Would trigger with 5% buffer, not 10%
            buffer_pct=5.0
        )

        assert should_transition is True
        assert target == PortfolioTier.SMALL


# ============================================================================
# Test: Transition Analysis
# ============================================================================


class TestTransitionAnalysis:
    """Test get_tier_transitions_needed method."""

    def test_no_transition_needed(self, tier_manager):
        """Test analysis when no transition is needed."""
        snapshot = PortfolioSnapshot(
            total_value=1500.0,
            cash=500.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        result = tier_manager.get_tier_transitions_needed(snapshot)

        assert result["transition_needed"] is False
        assert result["current_tier"] == "micro"
        assert result["target_tier"] is None
        assert len(result["changes_needed"]) == 0

    def test_transition_identifies_position_changes(self, tier_manager):
        """Test that transition identifies needed position changes."""
        snapshot = PortfolioSnapshot(
            total_value=3000.0,  # In small tier
            cash=500.0,
            positions=[PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95)],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,
            largest_position_pct=33.3,
            sector_exposures={"tech": 100.0},
        )

        result = tier_manager.get_tier_transitions_needed(snapshot)

        assert result["transition_needed"] is True
        assert result["target_tier"] == "small"
        # Should recommend increasing positions from 1 to 6
        assert any("increase positions" in change.lower() for change in result["changes_needed"])

    def test_transition_identifies_strategy_changes(self, tier_manager):
        """Test that transition identifies needed strategy changes."""
        snapshot = PortfolioSnapshot(
            total_value=3000.0,  # Moving to small tier
            cash=500.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        result = tier_manager.get_tier_transitions_needed(snapshot)

        # Should add mean_reversion strategy
        assert any("add strategies" in change.lower() and "mean_reversion" in change.lower()
                   for change in result["changes_needed"])

    def test_transition_includes_risk_limit_changes(self, tier_manager):
        """Test that transition includes risk limit changes."""
        snapshot = PortfolioSnapshot(
            total_value=15000.0,  # Triggers SMALL -> MEDIUM transition (needs >14000)
            cash=5000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.SMALL,
            positions_count=3,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        result = tier_manager.get_tier_transitions_needed(snapshot)

        # Should note change in daily risk limit (3% -> 4%)
        assert any("daily risk limit" in change.lower() for change in result["changes_needed"])


# ============================================================================
# Test: Tier Compatibility Validation
# ============================================================================


class TestTierCompatibility:
    """Test validate_tier_compatibility method."""

    def test_compatible_portfolio(self, tier_manager, portfolio_config):
        """Test validation of compatible portfolio."""
        tier_config = portfolio_config.tiers["micro"]

        snapshot = PortfolioSnapshot(
            total_value=1500.0,
            cash=500.0,
            positions=[
                PositionInfo("A", 10, 100, 100, 500, 0, 0, 1, 95),
                PositionInfo("B", 10, 100, 100, 500, 0, 0, 1, 95),
            ],
            daily_pnl=10.0,
            daily_pnl_pct=0.67,  # Below 2% limit
            quarterly_pnl_pct=5.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=2,
            largest_position_pct=20.0,
            sector_exposures={"tech": 50.0, "finance": 50.0},
        )

        is_compatible, issues = tier_manager.validate_tier_compatibility(
            tier_config, snapshot
        )

        assert is_compatible is True
        assert len(issues) == 0

    def test_too_many_positions(self, tier_manager, portfolio_config):
        """Test detection of too many positions."""
        tier_config = portfolio_config.tiers["micro"]  # Max 5 positions

        snapshot = PortfolioSnapshot(
            total_value=1500.0,
            cash=100.0,
            positions=[PositionInfo("A", 10, 100, 100, 200, 0, 0, 1, 95)] * 6,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=6,
            largest_position_pct=15.0,
            sector_exposures={"tech": 100.0},
        )

        is_compatible, issues = tier_manager.validate_tier_compatibility(
            tier_config, snapshot
        )

        assert is_compatible is False
        assert any("too many positions" in issue.lower() for issue in issues)

    def test_position_too_small(self, tier_manager, portfolio_config):
        """Test detection of positions below minimum size."""
        tier_config = portfolio_config.tiers["micro"]  # $100 minimum

        snapshot = PortfolioSnapshot(
            total_value=500.0,
            cash=400.0,
            positions=[PositionInfo("A", 10, 10, 10, 50, 0, 0, 1, 9.5)],  # $50 position
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,
            largest_position_pct=10.0,
            sector_exposures={"tech": 100.0},
        )

        is_compatible, issues = tier_manager.validate_tier_compatibility(
            tier_config, snapshot
        )

        assert is_compatible is False
        assert any("too small" in issue.lower() for issue in issues)

    def test_excessive_daily_pnl(self, tier_manager, portfolio_config):
        """Test detection of excessive daily P&L."""
        tier_config = portfolio_config.tiers["micro"]  # 2% daily limit

        snapshot = PortfolioSnapshot(
            total_value=1000.0,
            cash=500.0,
            positions=[PositionInfo("A", 10, 100, 100, 500, 0, 0, 1, 95)],
            daily_pnl=30.0,
            daily_pnl_pct=3.0,  # Exceeds 2% limit
            quarterly_pnl_pct=10.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,
            largest_position_pct=50.0,
            sector_exposures={"tech": 100.0},
        )

        is_compatible, issues = tier_manager.validate_tier_compatibility(
            tier_config, snapshot
        )

        assert is_compatible is False
        assert any("exceeds tier limit" in issue.lower() for issue in issues)

    def test_excessive_concentration(self, tier_manager, portfolio_config):
        """Test detection of excessive position concentration."""
        tier_config = portfolio_config.tiers["micro"]

        snapshot = PortfolioSnapshot(
            total_value=1000.0,
            cash=100.0,
            positions=[PositionInfo("A", 10, 100, 100, 900, 0, 0, 1, 95)],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,
            largest_position_pct=90.0,  # Excessive concentration
            sector_exposures={"tech": 100.0},
        )

        is_compatible, issues = tier_manager.validate_tier_compatibility(
            tier_config, snapshot
        )

        assert is_compatible is False
        assert any("concentrated" in issue.lower() for issue in issues)


# ============================================================================
# Test: Transition Step Recommendations
# ============================================================================


class TestTransitionStepRecommendations:
    """Test recommend_tier_transition_steps method."""

    def test_recommends_position_increases(self, tier_manager):
        """Test recommendation to increase positions."""
        snapshot = PortfolioSnapshot(
            total_value=5000.0,
            cash=4000.0,
            positions=[PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95)],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,
            largest_position_pct=20.0,
            sector_exposures={"tech": 100.0},
        )

        steps = tier_manager.recommend_tier_transition_steps(
            snapshot, PortfolioTier.SMALL
        )

        # Should recommend opening new positions (6 target - 1 current = 5 needed)
        assert any("open" in step.lower() and "position" in step.lower() for step in steps)

    def test_recommends_position_decreases(self, tier_manager):
        """Test recommendation to decrease positions."""
        snapshot = PortfolioSnapshot(
            total_value=50000.0,
            cash=10000.0,
            positions=[PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95)] * 12,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MEDIUM,
            positions_count=12,
            largest_position_pct=8.0,
            sector_exposures={"tech": 100.0},
        )

        steps = tier_manager.recommend_tier_transition_steps(
            snapshot, PortfolioTier.MICRO
        )

        # Should recommend closing positions (12 current - 5 max = 7 to close)
        assert any("close" in step.lower() and "position" in step.lower() for step in steps)

    def test_recommends_position_size_increases(self, tier_manager, portfolio_config):
        """Test recommendation to increase position sizes."""
        snapshot = PortfolioSnapshot(
            total_value=3000.0,
            cash=2000.0,
            positions=[
                PositionInfo("A", 10, 10, 10, 100, 0, 0, 1, 9.5),  # Too small for small tier
                PositionInfo("B", 10, 10, 10, 100, 0, 0, 1, 9.5),
            ],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=2,
            largest_position_pct=5.0,
            sector_exposures={"tech": 100.0},
        )

        steps = tier_manager.recommend_tier_transition_steps(
            snapshot, PortfolioTier.SMALL
        )

        # Should recommend increasing position sizes ($100 < $200 minimum)
        assert any("increase size" in step.lower() for step in steps)

    def test_recommends_new_strategies(self, tier_manager):
        """Test recommendation of new strategies."""
        snapshot = PortfolioSnapshot(
            total_value=11000.0,
            cash=1000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.SMALL,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        steps = tier_manager.recommend_tier_transition_steps(
            snapshot, PortfolioTier.MEDIUM
        )

        # Should recommend implementing new strategies (breakout)
        assert any("implement new strategies" in step.lower() or "breakout" in step.lower()
                   for step in steps)

    def test_recommends_risk_reduction(self, tier_manager):
        """Test recommendation to reduce risk when moving down tiers."""
        snapshot = PortfolioSnapshot(
            total_value=1500.0,
            cash=500.0,
            positions=[PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95)],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.SMALL,
            positions_count=1,
            largest_position_pct=66.7,
            sector_exposures={"tech": 100.0},
        )

        steps = tier_manager.recommend_tier_transition_steps(
            snapshot, PortfolioTier.MICRO
        )

        # Should recommend reducing position sizes for lower risk limits
        assert any("reduce position sizes" in step.lower() for step in steps)
