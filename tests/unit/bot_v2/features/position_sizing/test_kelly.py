"""
Comprehensive tests for Kelly Criterion position sizing.

Tests cover:
- Basic Kelly Criterion calculations
- Fractional Kelly sizing
- Edge cases and validation
- Position value conversions
- Risk metrics
- Drawdown protection
- Optimal fraction finding
- Growth simulation
"""

from decimal import Decimal

import pytest

from bot_v2.errors import RiskLimitExceeded, ValidationError
from bot_v2.features.position_sizing.kelly import (
    fractional_kelly,
    kelly_criterion,
    kelly_from_statistics,
    kelly_position_value,
    kelly_risk_metrics,
    kelly_with_drawdown_protection,
    optimal_kelly_fraction,
    simulate_kelly_growth,
    validate_kelly_inputs,
)
from bot_v2.features.position_sizing.types import RiskParameters, TradeStatistics


class TestKellyCriterionBasics:
    """Test basic Kelly Criterion calculations."""

    def test_kelly_criterion_positive_edge(self):
        """Test Kelly calculation with positive expected value."""
        # Win rate: 60%, Avg win: 5%, Avg loss: -3%
        kelly = kelly_criterion(0.6, 0.05, -0.03)

        # Expected: f* = p - q/b = 0.6 - 0.4/(0.05/0.03)
        # f* = 0.6 - 0.4/1.667 = 0.6 - 0.24 = 0.36
        assert 0.3 < kelly < 0.4
        assert kelly > 0

    def test_kelly_criterion_negative_edge(self):
        """Test Kelly returns zero for negative expected value."""
        # Win rate: 40%, Avg win: 3%, Avg loss: -5%
        kelly = kelly_criterion(0.4, 0.03, -0.05)

        # Negative expected value should return 0
        assert kelly == 0.0

    def test_kelly_criterion_fifty_fifty_equal_payoff(self):
        """Test 50/50 odds with equal payoff returns zero."""
        kelly = kelly_criterion(0.5, 0.03, -0.03)
        assert kelly == 0.0

    def test_kelly_criterion_very_high_edge(self):
        """Test Kelly with very high win rate caps extreme values."""
        # Very high win rate: 90%, good payoffs
        kelly = kelly_criterion(0.9, 0.1, -0.02)

        # Should cap extreme values
        assert kelly > 0
        assert kelly < 1.0

    def test_fractional_kelly_default(self):
        """Test fractional Kelly with default 1/4 fraction."""
        full_kelly = kelly_criterion(0.6, 0.05, -0.03)
        frac_kelly = fractional_kelly(0.6, 0.05, -0.03)

        assert abs(frac_kelly - full_kelly * 0.25) < 0.0001

    def test_fractional_kelly_custom_fraction(self):
        """Test fractional Kelly with custom fraction."""
        full_kelly = kelly_criterion(0.55, 0.04, -0.025)
        half_kelly = fractional_kelly(0.55, 0.04, -0.025, fraction=0.5)

        assert abs(half_kelly - full_kelly * 0.5) < 0.0001


class TestKellyValidation:
    """Test Kelly Criterion input validation."""

    def test_kelly_invalid_win_rate_too_low(self):
        """Test Kelly rejects win rate below 0.01."""
        with pytest.raises(ValidationError) as exc_info:
            kelly_criterion(0.005, 0.05, -0.03)
        assert "win_rate" in str(exc_info.value)

    def test_kelly_invalid_win_rate_too_high(self):
        """Test Kelly rejects win rate above 0.99."""
        with pytest.raises(ValidationError) as exc_info:
            kelly_criterion(0.995, 0.05, -0.03)
        assert "win_rate" in str(exc_info.value)

    def test_kelly_invalid_avg_win_negative(self):
        """Test Kelly rejects negative average win."""
        with pytest.raises(ValidationError) as exc_info:
            kelly_criterion(0.6, -0.05, -0.03)
        assert "avg_win" in str(exc_info.value)

    def test_kelly_invalid_avg_loss_positive(self):
        """Test Kelly rejects positive average loss."""
        with pytest.raises(ValidationError) as exc_info:
            kelly_criterion(0.6, 0.05, 0.03)
        assert "Average loss must be negative" in str(exc_info.value)

    def test_kelly_division_by_zero_protection(self):
        """Test Kelly handles near-zero average loss."""
        with pytest.raises(ValidationError) as exc_info:
            kelly_criterion(0.6, 0.05, -1e-11)
        assert "too close to zero" in str(exc_info.value).lower()

    def test_validate_kelly_inputs_valid(self):
        """Test validation passes for valid inputs."""
        errors = validate_kelly_inputs(0.6, 0.05, -0.03)
        assert len(errors) == 0

    def test_validate_kelly_inputs_invalid_win_rate(self):
        """Test validation catches invalid win rate."""
        errors = validate_kelly_inputs(1.5, 0.05, -0.03)
        assert len(errors) > 0
        assert any("win rate" in err.lower() for err in errors)

    def test_validate_kelly_inputs_negative_expected_value(self):
        """Test validation catches negative expected value."""
        errors = validate_kelly_inputs(0.3, 0.02, -0.05)
        assert len(errors) > 0
        assert any("expected value" in err.lower() for err in errors)


class TestKellyPositionValue:
    """Test conversion from Kelly fraction to position value."""

    def test_position_value_basic(self):
        """Test basic position value calculation."""
        risk_params = RiskParameters(
            max_position_size=0.2, min_position_size=0.01, kelly_fraction=0.25
        )

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.15,
            price_per_share=100.0,
            risk_params=risk_params,
        )

        # 10000 * 0.15 = 1500, at $100/share = 15 shares
        assert shares == 15
        assert value == 1500.0

    def test_position_value_respects_max_limit(self):
        """Test position value caps at max limit."""
        risk_params = RiskParameters(
            max_position_size=0.1, min_position_size=0.01, kelly_fraction=0.25
        )

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.5,  # Would want 50% but max is 10%
            price_per_share=100.0,
            risk_params=risk_params,
        )

        # Should cap at 10% = $1000 = 10 shares
        assert shares == 10
        assert value == 1000.0

    def test_position_value_respects_min_limit(self):
        """Test position value floors at min limit."""
        risk_params = RiskParameters(
            max_position_size=0.2, min_position_size=0.05, kelly_fraction=0.25
        )

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.01,  # Would want 1% but min is 5%
            price_per_share=100.0,
            risk_params=risk_params,
        )

        # Should floor at 5% = $500 = 5 shares
        assert shares == 5
        assert value == 500.0

    def test_position_value_expensive_shares(self):
        """Test position value with expensive shares."""
        risk_params = RiskParameters(
            max_position_size=0.2, min_position_size=0.01, kelly_fraction=0.25
        )

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.15,  # Want $1500
            price_per_share=5000.0,  # But shares are $5000 each
            risk_params=risk_params,
        )

        # Can't afford any shares at min position size
        assert shares == 0
        assert value == 0.0

    def test_position_value_fractional_shares_rounds_down(self):
        """Test that fractional shares are rounded down."""
        risk_params = RiskParameters(
            max_position_size=0.2, min_position_size=0.01, kelly_fraction=0.25
        )

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.12,  # Want $1200
            price_per_share=75.0,  # Would be 16 shares exactly
            risk_params=risk_params,
        )

        assert shares == 16
        assert value == 1200.0

    def test_position_value_exceeds_portfolio(self):
        """Test position gets capped when Kelly would exceed max limit."""
        risk_params = RiskParameters(
            max_position_size=0.5,  # Max 50% of portfolio
            min_position_size=0.01,
            kelly_fraction=0.25,
        )

        value, shares = kelly_position_value(
            portfolio_value=1000.0,
            kelly_fraction=0.9,  # Want 90% of portfolio (exceeds 50% max)
            price_per_share=10.0,
            risk_params=risk_params,
        )

        # Should be capped at max_position_size (50% = $500)
        assert value <= 500.0
        assert shares > 0  # Should still return valid shares within cap

    def test_position_value_invalid_inputs(self):
        """Test validation of position value inputs."""
        risk_params = RiskParameters()

        with pytest.raises(ValidationError):
            kelly_position_value(
                portfolio_value=-1000.0,  # Negative portfolio
                kelly_fraction=0.1,
                price_per_share=100.0,
                risk_params=risk_params,
            )

        with pytest.raises(ValidationError):
            kelly_position_value(
                portfolio_value=1000.0,
                kelly_fraction=1.5,  # Kelly > 1.0
                price_per_share=100.0,
                risk_params=risk_params,
            )


class TestKellyRiskMetrics:
    """Test Kelly risk metric calculations."""

    def test_risk_metrics_basic(self):
        """Test basic risk metrics calculation."""
        metrics = kelly_risk_metrics(
            kelly_fraction=0.2, avg_loss=-0.05, portfolio_value=10000.0
        )

        assert metrics["kelly_fraction"] == 0.2
        assert metrics["position_value"] == 2000.0
        assert metrics["max_expected_loss"] == 100.0  # 2000 * 0.05
        assert metrics["max_loss_pct"] == 0.01  # 100/10000

    def test_risk_metrics_large_loss(self):
        """Test risk metrics with large average loss."""
        metrics = kelly_risk_metrics(
            kelly_fraction=0.15, avg_loss=-0.15, portfolio_value=50000.0
        )

        assert metrics["position_value"] == 7500.0
        assert metrics["max_expected_loss"] == 1125.0  # 7500 * 0.15
        assert abs(metrics["max_loss_pct"] - 0.0225) < 0.0001


class TestKellyFromStatistics:
    """Test Kelly calculation from trade statistics."""

    def test_kelly_from_statistics_sufficient_data(self):
        """Test Kelly calculation with sufficient trade history."""
        stats = TradeStatistics(
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            total_return=0.5,
            avg_win_return=0.06,
            avg_loss_return=-0.04,
            win_rate=0.6,
            profit_factor=1.8,
        )

        kelly = kelly_from_statistics(stats, fraction=0.25)
        assert kelly > 0
        assert kelly < 0.5

    def test_kelly_from_statistics_insufficient_data(self):
        """Test Kelly returns zero with insufficient trades."""
        stats = TradeStatistics(
            total_trades=5,  # Less than minimum 10
            winning_trades=3,
            losing_trades=2,
            total_return=0.1,
            avg_win_return=0.05,
            avg_loss_return=-0.03,
            win_rate=0.6,
            profit_factor=1.5,
        )

        kelly = kelly_from_statistics(stats, fraction=0.25)
        assert kelly == 0.0


class TestKellyDrawdownProtection:
    """Test Kelly with drawdown protection."""

    def test_drawdown_no_reduction_below_threshold(self):
        """Test no reduction when drawdown below threshold."""
        base_kelly = fractional_kelly(0.6, 0.05, -0.03, 0.25)
        protected_kelly = kelly_with_drawdown_protection(
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            current_drawdown=0.05,  # Below 10% threshold
            max_drawdown_threshold=0.1,
        )

        assert protected_kelly == base_kelly

    def test_drawdown_reduction_above_threshold(self):
        """Test reduction when drawdown exceeds threshold."""
        base_kelly = fractional_kelly(0.6, 0.05, -0.03, 0.25)
        protected_kelly = kelly_with_drawdown_protection(
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            current_drawdown=0.2,  # 20% drawdown
            max_drawdown_threshold=0.1,
        )

        assert protected_kelly < base_kelly
        assert protected_kelly > 0

    def test_drawdown_severe_reduction(self):
        """Test severe reduction with large drawdown."""
        protected_kelly = kelly_with_drawdown_protection(
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            current_drawdown=0.4,  # 40% drawdown
            max_drawdown_threshold=0.1,
        )

        # Should be significantly reduced
        assert protected_kelly < 0.05


class TestOptimalKellyFraction:
    """Test optimal Kelly fraction finding."""

    def test_optimal_fraction_positive_returns(self):
        """Test finding optimal fraction with profitable strategy."""
        # Simulate profitable strategy returns
        returns = [0.05, 0.03, -0.02, 0.04, 0.06, -0.01, 0.03, 0.02, -0.02, 0.05, 0.04, -0.01]

        fraction, wealth = optimal_kelly_fraction(returns)

        assert 0 < fraction <= 1.0
        assert wealth >= 1.0  # Should grow

    def test_optimal_fraction_insufficient_data(self):
        """Test returns default with insufficient data."""
        returns = [0.01, -0.01, 0.02]  # Too few

        fraction, wealth = optimal_kelly_fraction(returns)

        assert fraction == 0.0
        assert wealth == 1.0

    def test_optimal_fraction_losing_strategy(self):
        """Test returns zero for losing strategy."""
        returns = [-0.05, -0.03, 0.01, -0.04, -0.02, -0.01, -0.03, 0.02, -0.04, -0.05]

        fraction, wealth = optimal_kelly_fraction(returns)

        assert fraction == 0.0

    def test_optimal_fraction_custom_test_fractions(self):
        """Test with custom test fractions."""
        returns = [0.05, 0.03, -0.02, 0.04, 0.06, -0.01, 0.03, 0.02, -0.02, 0.05, 0.04, -0.01]

        fraction, wealth = optimal_kelly_fraction(returns, test_fractions=[0.1, 0.2, 0.3])

        assert fraction in [0.1, 0.2, 0.3]


class TestSimulateKellyGrowth:
    """Test Kelly growth simulation."""

    def test_simulate_growth_profitable(self):
        """Test growth simulation with profitable returns."""
        returns = [0.05, 0.03, -0.02, 0.04, 0.06, -0.01, 0.03]

        final_wealth = simulate_kelly_growth(returns, kelly_fraction=0.25, initial_wealth=1.0)

        assert final_wealth > 1.0

    def test_simulate_growth_losing(self):
        """Test growth simulation with losing returns."""
        returns = [-0.05, -0.03, -0.02, -0.04]

        final_wealth = simulate_kelly_growth(returns, kelly_fraction=0.25, initial_wealth=1.0)

        assert final_wealth < 1.0

    def test_simulate_growth_bankruptcy_protection(self):
        """Test bankruptcy protection in simulation."""
        returns = [-0.5, -0.5, -0.5, -0.5]  # Severe losses

        final_wealth = simulate_kelly_growth(returns, kelly_fraction=0.9, initial_wealth=1.0)

        # Should never go below 0.01
        assert final_wealth >= 0.01

    def test_simulate_growth_zero_kelly(self):
        """Test simulation with zero Kelly fraction."""
        returns = [0.05, 0.03, -0.02, 0.04]

        final_wealth = simulate_kelly_growth(returns, kelly_fraction=0.0, initial_wealth=1.0)

        # No position = no change
        assert final_wealth == 1.0

    def test_simulate_growth_custom_initial_wealth(self):
        """Test simulation with custom initial wealth."""
        returns = [0.1, 0.05, 0.08]

        final_wealth = simulate_kelly_growth(returns, kelly_fraction=0.25, initial_wealth=10000.0)

        assert final_wealth > 10000.0


class TestKellyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_kelly_extreme_odds_ratio_capped(self):
        """Test that extreme odds ratios are capped."""
        # Very high win, very small loss
        kelly = kelly_criterion(0.7, 0.5, -0.001)

        # Should cap extreme values
        assert kelly >= 0
        assert kelly < 1.0

    def test_kelly_boundary_win_rates(self):
        """Test Kelly at boundary win rates."""
        # Just above minimum
        kelly_low = kelly_criterion(0.011, 0.05, -0.03)
        assert kelly_low >= 0

        # Just below maximum
        kelly_high = kelly_criterion(0.989, 0.05, -0.03)
        assert kelly_high >= 0

    def test_fractional_kelly_zero_fraction(self):
        """Test fractional Kelly with zero fraction."""
        kelly = fractional_kelly(0.6, 0.05, -0.03, fraction=0.0)
        assert kelly == 0.0

    def test_fractional_kelly_full_kelly(self):
        """Test fractional Kelly with 1.0 fraction."""
        full = kelly_criterion(0.6, 0.05, -0.03)
        frac = fractional_kelly(0.6, 0.05, -0.03, fraction=1.0)
        assert abs(full - frac) < 0.0001

    def test_position_value_zero_kelly(self):
        """Test position value with zero Kelly fraction returns minimal position."""
        risk_params = RiskParameters()

        value, shares = kelly_position_value(
            portfolio_value=10000.0,
            kelly_fraction=0.0,
            price_per_share=100.0,
            risk_params=risk_params,
        )

        # With Kelly=0, implementation returns min position (1 share)
        assert shares == 1
        assert value == 100.0

    def test_position_value_tiny_portfolio(self):
        """Test position value with very small portfolio."""
        risk_params = RiskParameters(max_position_size=0.1, min_position_size=0.01)

        value, shares = kelly_position_value(
            portfolio_value=10.0,  # Only $10
            kelly_fraction=0.1,
            price_per_share=5.0,
            risk_params=risk_params,
        )

        # $10 * 0.1 = $1, but min is $0.1, so 0 shares at $5/share
        assert shares == 0
        assert value == 0.0