"""
Comprehensive tests for AdaptiveRiskManager.

Tests cover:
- Risk metrics calculation (concentration, variance, drawdown)
- Position size limit enforcement
- Position size calculation with confidence scaling
- Trading frequency limits and PDT safeguards
- Stop loss price calculation (long/short)
- Portfolio risk level assessment
- Tier compliance checking
- Helper methods (HHI, variance, etc.)
"""

import pytest

from bot_v2.features.adaptive_portfolio.risk_manager import AdaptiveRiskManager
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
    """Create test portfolio configuration."""
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

    return PortfolioConfig(
        version="1.0",
        last_updated="2025-01-01",
        description="Test config",
        tiers={"micro": micro_tier, "small": small_tier},
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
def risk_manager(portfolio_config):
    """Create AdaptiveRiskManager instance."""
    return AdaptiveRiskManager(portfolio_config)


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return [
        PositionInfo(
            symbol="AAPL",
            shares=10,
            entry_price=150.0,
            current_price=160.0,
            position_value=1600.0,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=6.67,
            days_held=5,
            stop_loss_price=142.5,
        ),
        PositionInfo(
            symbol="MSFT",
            shares=5,
            entry_price=300.0,
            current_price=310.0,
            position_value=1550.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=3.33,
            days_held=3,
            stop_loss_price=285.0,
        ),
        PositionInfo(
            symbol="GOOGL",
            shares=8,
            entry_price=125.0,
            current_price=130.0,
            position_value=1040.0,
            unrealized_pnl=40.0,
            unrealized_pnl_pct=4.0,
            days_held=7,
            stop_loss_price=118.75,
        ),
    ]


@pytest.fixture
def portfolio_snapshot(sample_positions):
    """Create sample portfolio snapshot."""
    return PortfolioSnapshot(
        total_value=5000.0,
        cash=810.0,
        positions=sample_positions,
        daily_pnl=50.0,
        daily_pnl_pct=1.0,
        quarterly_pnl_pct=5.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=3,
        largest_position_pct=32.0,  # 1600/5000
        sector_exposures={"tech": 100.0},
    )


# ============================================================================
# Test: Risk Metrics Calculation
# ============================================================================


class TestRiskMetricsCalculation:
    """Test calculate_risk_metrics method."""

    def test_calculates_basic_metrics(self, risk_manager, portfolio_snapshot, portfolio_config):
        """Test calculation of basic portfolio metrics."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert metrics["total_value"] == 5000.0
        assert metrics["cash_pct"] == pytest.approx(16.2, abs=0.1)  # 810/5000*100
        assert metrics["invested_pct"] == pytest.approx(83.8, abs=0.1)
        assert metrics["positions_count"] == 3

    def test_calculates_concentration_risk(
        self, risk_manager, portfolio_snapshot, portfolio_config
    ):
        """Test position concentration risk calculation."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert "position_concentration_risk" in metrics
        assert 0 <= metrics["position_concentration_risk"] <= 100

    def test_calculates_daily_risk_metrics(
        self, risk_manager, portfolio_snapshot, portfolio_config
    ):
        """Test daily risk and utilization metrics."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert metrics["daily_pnl_pct"] == 1.0
        assert metrics["daily_risk_pct"] == 1.0
        assert metrics["daily_risk_limit_pct"] == 2.0
        assert metrics["daily_risk_utilization_pct"] == 50.0  # 1.0/2.0*100

    def test_calculates_position_sizing_metrics(
        self, risk_manager, portfolio_snapshot, portfolio_config
    ):
        """Test position sizing variance calculations."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert "avg_position_size" in metrics
        assert "position_size_variance" in metrics
        assert metrics["avg_position_size"] > 0

    def test_calculates_tier_compliance(self, risk_manager, portfolio_snapshot, portfolio_config):
        """Test tier compliance calculation."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert "tier_compliant" in metrics
        assert isinstance(metrics["tier_compliant"], bool)

    def test_calculates_risk_adjusted_score(
        self, risk_manager, portfolio_snapshot, portfolio_config
    ):
        """Test risk-adjusted score calculation."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert "risk_adjusted_score" in metrics
        assert 0 <= metrics["risk_adjusted_score"] <= 100

    def test_estimates_max_drawdown(self, risk_manager, portfolio_snapshot, portfolio_config):
        """Test max drawdown estimation."""
        tier_config = portfolio_config.tiers["micro"]

        metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)

        assert "estimated_max_drawdown_pct" in metrics
        assert 0 <= metrics["estimated_max_drawdown_pct"] <= 50


# ============================================================================
# Test: Position Size Limits
# ============================================================================


class TestPositionSizeLimits:
    """Test check_position_size_limits method."""

    def test_accepts_valid_position_size(self, risk_manager, portfolio_config):
        """Test that valid position sizes are accepted."""
        tier_config = portfolio_config.tiers["micro"]

        is_valid, reason = risk_manager.check_position_size_limits(
            position_value=500.0, total_portfolio_value=5000.0, tier_config=tier_config
        )

        assert is_valid is True
        assert "acceptable" in reason.lower()

    def test_rejects_position_below_minimum(self, risk_manager, portfolio_config):
        """Test rejection of positions below minimum size."""
        tier_config = portfolio_config.tiers["micro"]

        is_valid, reason = risk_manager.check_position_size_limits(
            position_value=50.0,  # Below $100 minimum
            total_portfolio_value=5000.0,
            tier_config=tier_config,
        )

        assert is_valid is False
        assert "too small" in reason.lower()
        assert "100" in reason

    def test_rejects_position_above_maximum_pct(self, risk_manager, portfolio_config):
        """Test rejection of positions exceeding max percentage."""
        tier_config = portfolio_config.tiers["micro"]

        is_valid, reason = risk_manager.check_position_size_limits(
            position_value=1500.0,  # 30% of portfolio (>25% max)
            total_portfolio_value=5000.0,
            tier_config=tier_config,
        )

        assert is_valid is False
        assert "too large" in reason.lower()
        assert "25" in reason


# ============================================================================
# Test: Position Size Calculation
# ============================================================================


class TestPositionSizeCalculation:
    """Test calculate_position_size method."""

    def test_calculates_base_position_size(self, risk_manager, portfolio_config):
        """Test base position size calculation."""
        tier_config = portfolio_config.tiers["micro"]

        size = risk_manager.calculate_position_size(
            total_portfolio_value=10000.0, tier_config=tier_config, confidence=1.0
        )

        # For 3 target positions: 10000/3 = 3333, capped at 25% = 2500
        assert size == pytest.approx(2500.0, abs=1.0)

    def test_scales_with_confidence(self, risk_manager, portfolio_config):
        """Test that position size scales with confidence."""
        tier_config = portfolio_config.tiers["micro"]

        size_high = risk_manager.calculate_position_size(
            total_portfolio_value=10000.0, tier_config=tier_config, confidence=1.0
        )

        size_low = risk_manager.calculate_position_size(
            total_portfolio_value=10000.0, tier_config=tier_config, confidence=0.5
        )

        # Low confidence should yield smaller position
        assert size_low < size_high

    def test_respects_minimum_position_size(self, risk_manager, portfolio_config):
        """Test that minimum position size is enforced before small portfolio adjustment."""
        tier_config = portfolio_config.tiers["micro"]

        size = risk_manager.calculate_position_size(
            total_portfolio_value=1000.0,  # Small portfolio
            tier_config=tier_config,
            confidence=0.1,  # Very low confidence
        )

        # With small portfolio (<5000), 0.8 multiplier applies after other checks
        # So minimum is enforced but then reduced by 0.8
        expected_min = tier_config.min_position_size * 0.8
        assert size == pytest.approx(expected_min, abs=1.0)

    def test_respects_maximum_position_percentage(self, risk_manager, portfolio_config):
        """Test that maximum position percentage is enforced."""
        tier_config = portfolio_config.tiers["micro"]

        size = risk_manager.calculate_position_size(
            total_portfolio_value=1000.0, tier_config=tier_config, confidence=1.0
        )

        # Should not exceed 25% of portfolio
        max_allowed = 1000.0 * 0.25
        assert size <= max_allowed

    def test_applies_conservative_adjustment_for_small_portfolios(
        self, risk_manager, portfolio_config
    ):
        """Test conservative adjustment for small portfolios."""
        tier_config = portfolio_config.tiers["micro"]

        size_small = risk_manager.calculate_position_size(
            total_portfolio_value=3000.0,  # < 5000 threshold
            tier_config=tier_config,
            confidence=1.0,
        )

        size_large = risk_manager.calculate_position_size(
            total_portfolio_value=6000.0,  # > 5000 threshold
            tier_config=tier_config,
            confidence=1.0,
        )

        # Small portfolio should have smaller positions (80% factor)
        expected_ratio = (3000.0 / 3) * 0.8 / (6000.0 / 3)
        actual_ratio = size_small / size_large

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.05)


# ============================================================================
# Test: Trading Frequency Limits
# ============================================================================


class TestTradingFrequencyLimits:
    """Test check_trading_frequency_limits method."""

    def test_allows_trading_below_limit(self, risk_manager, portfolio_config):
        """Test trading is allowed below weekly limit (when not near PDT limit)."""
        tier_config = portfolio_config.tiers["micro"]

        can_trade, reason = risk_manager.check_trading_frequency_limits(
            trades_this_week=1, tier_config=tier_config  # Below PDT threshold of 2
        )

        assert can_trade is True
        assert "can trade" in reason.lower()

    def test_blocks_trading_at_limit(self, risk_manager, portfolio_config):
        """Test trading is blocked at weekly limit."""
        tier_config = portfolio_config.tiers["micro"]

        can_trade, reason = risk_manager.check_trading_frequency_limits(
            trades_this_week=5, tier_config=tier_config  # At the 5 trade limit
        )

        assert can_trade is False
        assert "limit reached" in reason.lower()

    def test_pdt_check_for_compliant_accounts(self, risk_manager, portfolio_config):
        """Test PDT check blocks trades for compliant accounts."""
        tier_config = portfolio_config.tiers["micro"]  # PDT compliant

        can_trade, reason = risk_manager.check_trading_frequency_limits(
            trades_this_week=2, tier_config=tier_config  # Approaching 3 day trade limit
        )

        assert can_trade is False
        assert "pdt" in reason.lower()

    def test_no_pdt_check_for_non_compliant_accounts(self, risk_manager, portfolio_config):
        """Test no PDT check for non-compliant accounts."""
        tier_config = portfolio_config.tiers["small"]  # Not PDT compliant

        can_trade, reason = risk_manager.check_trading_frequency_limits(
            trades_this_week=2, tier_config=tier_config
        )

        assert can_trade is True  # Should allow trade


# ============================================================================
# Test: Stop Loss Calculation
# ============================================================================


class TestStopLossCalculation:
    """Test calculate_stop_loss_price method."""

    def test_calculates_stop_loss_for_long_position(self, risk_manager, portfolio_config):
        """Test stop loss calculation for long positions."""
        tier_config = portfolio_config.tiers["micro"]

        stop_loss = risk_manager.calculate_stop_loss_price(
            entry_price=100.0, tier_config=tier_config, position_direction="LONG"
        )

        # 5% stop loss for micro tier
        expected = 100.0 * (1 - 0.05)
        assert stop_loss == pytest.approx(expected, abs=0.01)

    def test_calculates_stop_loss_for_short_position(self, risk_manager, portfolio_config):
        """Test stop loss calculation for short positions."""
        tier_config = portfolio_config.tiers["micro"]

        stop_loss = risk_manager.calculate_stop_loss_price(
            entry_price=100.0, tier_config=tier_config, position_direction="SHORT"
        )

        # 5% stop loss for micro tier (above entry for shorts)
        expected = 100.0 * (1 + 0.05)
        assert stop_loss == pytest.approx(expected, abs=0.01)

    def test_different_stop_loss_for_different_tiers(self, risk_manager, portfolio_config):
        """Test different tiers have different stop losses."""
        micro_config = portfolio_config.tiers["micro"]
        small_config = portfolio_config.tiers["small"]

        stop_micro = risk_manager.calculate_stop_loss_price(
            entry_price=100.0, tier_config=micro_config, position_direction="LONG"
        )

        stop_small = risk_manager.calculate_stop_loss_price(
            entry_price=100.0, tier_config=small_config, position_direction="LONG"
        )

        # Micro: 5% stop, Small: 7% stop
        assert stop_micro > stop_small  # Tighter stop for micro


# ============================================================================
# Test: Portfolio Risk Assessment
# ============================================================================


class TestPortfolioRiskAssessment:
    """Test assess_portfolio_risk_level method."""

    def test_assesses_low_risk_portfolio(self, risk_manager, portfolio_config):
        """Test assessment of low-risk portfolio."""
        tier_config = portfolio_config.tiers["micro"]

        # Create conservative portfolio
        conservative_snapshot = PortfolioSnapshot(
            total_value=5000.0,
            cash=2500.0,  # 50% cash
            positions=[
                PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95),
                PositionInfo("B", 10, 100, 100, 1000, 0, 0, 1, 95),
                PositionInfo("C", 10, 100, 100, 500, 0, 0, 1, 95),
            ],
            daily_pnl=10.0,
            daily_pnl_pct=0.2,  # Low daily risk
            quarterly_pnl_pct=2.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=3,
            largest_position_pct=20.0,  # Moderate concentration
            sector_exposures={"tech": 50.0, "finance": 30.0, "health": 20.0},
        )

        risk_level, risk_score, risk_factors = risk_manager.assess_portfolio_risk_level(
            conservative_snapshot, tier_config
        )

        assert risk_level in ["VERY_LOW", "LOW"]
        assert risk_score < 4

    def test_assesses_high_risk_portfolio(self, risk_manager, portfolio_config):
        """Test assessment of high-risk portfolio."""
        tier_config = portfolio_config.tiers["micro"]

        # Create aggressive portfolio
        aggressive_snapshot = PortfolioSnapshot(
            total_value=2100.0,  # Near tier minimum
            cash=100.0,  # Very low cash
            positions=[
                PositionInfo("A", 10, 100, 100, 2000, 0, 0, 1, 95),  # Very large position
            ],
            daily_pnl=-40.0,
            daily_pnl_pct=-1.9,  # High daily risk (95% of 2% limit)
            quarterly_pnl_pct=-8.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=1,  # Too few positions
            largest_position_pct=95.0,  # Extreme concentration
            sector_exposures={"tech": 100.0},
        )

        risk_level, risk_score, risk_factors = risk_manager.assess_portfolio_risk_level(
            aggressive_snapshot, tier_config
        )

        assert risk_level in ["MEDIUM", "HIGH"]
        assert risk_score >= 4
        assert len(risk_factors) > 0

    def test_identifies_specific_risk_factors(
        self, risk_manager, portfolio_snapshot, portfolio_config
    ):
        """Test identification of specific risk factors."""
        tier_config = portfolio_config.tiers["micro"]

        _, _, risk_factors = risk_manager.assess_portfolio_risk_level(
            portfolio_snapshot, tier_config
        )

        # Should identify high concentration (32%)
        assert any("concentration" in factor.lower() for factor in risk_factors)


# ============================================================================
# Test: Helper Methods
# ============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_concentration_risk_single_position(self, risk_manager):
        """Test concentration risk for single position returns 0 (edge case)."""
        positions = [PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95)]

        concentration = risk_manager._calculate_concentration_risk(positions)

        # Single position: max_hhi == min_hhi, returns 0
        assert concentration == pytest.approx(0, abs=1.0)

    def test_concentration_risk_equal_positions(self, risk_manager):
        """Test concentration risk for equally weighted positions (minimum concentration)."""
        positions = [
            PositionInfo("A", 10, 100, 100, 1000, 0, 0, 1, 95),
            PositionInfo("B", 10, 100, 100, 1000, 0, 0, 1, 95),
            PositionInfo("C", 10, 100, 100, 1000, 0, 0, 1, 95),
        ]

        concentration = risk_manager._calculate_concentration_risk(positions)

        # Equal weights = minimum concentration
        assert concentration == pytest.approx(0, abs=1.0)

    def test_concentration_risk_empty_positions(self, risk_manager):
        """Test concentration risk for empty portfolio."""
        concentration = risk_manager._calculate_concentration_risk([])

        assert concentration == 0

    def test_avg_position_size(self, risk_manager, sample_positions):
        """Test average position size calculation."""
        total_value = 5000.0

        avg_size = risk_manager._calculate_avg_position_size(sample_positions, total_value)

        # Total invested: 4190, 3 positions, portfolio 5000
        # Avg = 4190/3/5000*100 = ~27.9%
        assert avg_size == pytest.approx(27.9, abs=1.0)

    def test_position_size_variance(self, risk_manager, sample_positions):
        """Test position size variance calculation."""
        total_value = 5000.0

        variance = risk_manager._calculate_position_size_variance(sample_positions, total_value)

        assert variance > 0  # Should have some variance

    def test_tier_compliance_check(self, risk_manager, portfolio_snapshot, portfolio_config):
        """Test tier compliance checking."""
        tier_config = portfolio_config.tiers["micro"]

        is_compliant = risk_manager._check_tier_compliance(portfolio_snapshot, tier_config)

        # Should be compliant (3 positions <= 5 max, sizes >= 100, daily risk <= 2%)
        assert is_compliant is True

    def test_tier_compliance_fails_for_too_many_positions(self, risk_manager, portfolio_config):
        """Test compliance fails with too many positions."""
        tier_config = portfolio_config.tiers["micro"]  # Max 5 positions

        snapshot = PortfolioSnapshot(
            total_value=5000.0,
            cash=1000.0,
            positions=[PositionInfo("A", 10, 100, 100, 400, 0, 0, 1, 95)] * 6,  # 6 positions
            daily_pnl=10.0,
            daily_pnl_pct=0.2,
            quarterly_pnl_pct=2.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=6,
            largest_position_pct=10.0,
            sector_exposures={"tech": 100.0},
        )

        is_compliant = risk_manager._check_tier_compliance(snapshot, tier_config)

        assert is_compliant is False
