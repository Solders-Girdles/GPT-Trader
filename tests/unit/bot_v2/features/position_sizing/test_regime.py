"""
Comprehensive tests for regime-based position sizing.

Tests cover:
- Regime adjustments for different market conditions
- Dynamic regime multiplier calculation
- Regime transitions and confidence
- Momentum factors
- Portfolio allocation across regimes
- Correlation adjustments
- Volatility scaling
- Input validation
"""

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.position_sizing.regime import (
    dynamic_regime_multipliers,
    portfolio_regime_allocation,
    regime_adjusted_size,
    regime_correlation_adjustment,
    regime_momentum_factor,
    regime_transition_adjustment,
    regime_volatility_scaling,
    safe_regime_calculation,
    validate_regime_inputs,
)
from bot_v2.features.position_sizing.types import RegimeMultipliers


class TestRegimeAdjustment:
    """Test basic regime adjustment functionality."""

    def test_regime_bull_quiet_increases_position(self):
        """Test bull quiet regime increases position size."""
        multipliers = RegimeMultipliers()

        adjusted, explanation = regime_adjusted_size(0.1, "bull_quiet", multipliers)

        assert adjusted > 0.1  # 1.2x multiplier
        assert "bull_quiet" in explanation

    def test_regime_bear_volatile_decreases_position(self):
        """Test bear volatile regime decreases position size."""
        multipliers = RegimeMultipliers()

        adjusted, explanation = regime_adjusted_size(0.1, "bear_volatile", multipliers)

        assert adjusted < 0.1  # 0.4x multiplier
        assert "bear_volatile" in explanation

    def test_regime_crisis_severe_reduction(self):
        """Test crisis regime severely reduces position."""
        multipliers = RegimeMultipliers()

        adjusted, explanation = regime_adjusted_size(0.1, "crisis", multipliers)

        assert adjusted <= 0.025  # 0.2x multiplier on 0.1
        assert "crisis" in explanation

    def test_regime_sideways_quiet_neutral(self):
        """Test sideways quiet regime keeps position neutral."""
        multipliers = RegimeMultipliers()

        adjusted, explanation = regime_adjusted_size(0.1, "sideways_quiet", multipliers)

        assert abs(adjusted - 0.1) < 0.01  # 1.0x multiplier
        assert "sideways_quiet" in explanation

    def test_regime_adjustment_all_regimes(self):
        """Test adjustment works for all valid regimes."""
        multipliers = RegimeMultipliers()
        base_size = 0.15

        regimes = [
            "bull_quiet",
            "bull_volatile",
            "bear_quiet",
            "bear_volatile",
            "sideways_quiet",
            "sideways_volatile",
            "crisis",
        ]

        for regime in regimes:
            adjusted, explanation = regime_adjusted_size(base_size, regime, multipliers)
            assert adjusted >= 0
            assert regime in explanation

    def test_regime_custom_multipliers(self):
        """Test regime adjustment with custom multipliers."""
        custom_multipliers = RegimeMultipliers(bull_quiet=2.0, bear_volatile=0.1, crisis=0.05)

        bull_adjusted, _ = regime_adjusted_size(0.1, "bull_quiet", custom_multipliers)
        assert abs(bull_adjusted - 0.2) < 1e-10  # 2.0x (use approx for floating point)

        bear_adjusted, _ = regime_adjusted_size(0.1, "bear_volatile", custom_multipliers)
        assert abs(bear_adjusted - 0.01) < 1e-10  # 0.1x (use approx for floating point)

    def test_regime_adjustment_caps_extreme_values(self):
        """Test extreme multipliers are capped."""
        extreme_multipliers = RegimeMultipliers()
        extreme_multipliers.bull_quiet = 10.0  # Unrealistic

        adjusted, _ = regime_adjusted_size(0.5, "bull_quiet", extreme_multipliers)

        # Should be capped to reasonable bounds (max 1.0 for adjusted size)
        assert adjusted <= 1.0

    def test_regime_no_regime_data(self):
        """Test handling when no regime data provided."""
        multipliers = RegimeMultipliers()

        adjusted, explanation = regime_adjusted_size(0.1, "", multipliers)

        assert adjusted == 0.1  # No change
        assert "No regime data" in explanation or "no regime" in explanation.lower()

    def test_regime_zero_base_size(self):
        """Test regime adjustment with zero base size."""
        multipliers = RegimeMultipliers()

        adjusted, _ = regime_adjusted_size(0.0, "bull_quiet", multipliers)
        assert adjusted == 0.0


class TestRegimeValidation:
    """Test regime input validation."""

    def test_validate_valid_regime(self):
        """Test validation passes for valid regime."""
        multipliers = RegimeMultipliers()

        errors = validate_regime_inputs("bull_quiet", multipliers)
        assert len(errors) == 0

    def test_validate_invalid_regime(self):
        """Test validation catches invalid regime."""
        multipliers = RegimeMultipliers()

        with pytest.raises(ValidationError):
            regime_adjusted_size(0.1, "invalid_regime", multipliers)

    def test_validate_extreme_multipliers(self):
        """Test validation catches extreme multipliers."""
        multipliers = RegimeMultipliers(bull_quiet=10.0)  # Beyond reasonable

        errors = validate_regime_inputs("bull_quiet", multipliers)
        assert len(errors) > 0

    def test_validate_negative_multipliers(self):
        """Test validation catches negative multipliers."""
        multipliers = RegimeMultipliers(bear_quiet=-0.5)

        errors = validate_regime_inputs("bear_quiet", multipliers)
        assert len(errors) > 0

    def test_validate_all_regimes(self):
        """Test validation works for all regime types."""
        multipliers = RegimeMultipliers()

        regimes = [
            "bull_quiet",
            "bull_volatile",
            "bear_quiet",
            "bear_volatile",
            "sideways_quiet",
            "sideways_volatile",
            "crisis",
        ]

        for regime in regimes:
            errors = validate_regime_inputs(regime, multipliers)
            assert len(errors) == 0


class TestDynamicRegimeMultipliers:
    """Test dynamic regime multiplier calculation."""

    def test_dynamic_multipliers_positive_performance(self):
        """Test multipliers increase with positive regime performance."""
        regime_history = [
            ("bull_quiet", 0.05),
            ("bull_quiet", 0.08),
            ("bull_quiet", 0.06),
            ("bull_quiet", 0.07),
        ]

        multipliers = dynamic_regime_multipliers(regime_history)

        # Bull quiet should have increased multiplier
        assert multipliers.bull_quiet >= RegimeMultipliers().bull_quiet

    def test_dynamic_multipliers_negative_performance(self):
        """Test multipliers decrease with negative regime performance."""
        regime_history = [
            ("bear_volatile", -0.05),
            ("bear_volatile", -0.08),
            ("bear_volatile", -0.06),
        ]

        multipliers = dynamic_regime_multipliers(regime_history)

        # Bear volatile should have decreased multiplier
        assert multipliers.bear_volatile <= RegimeMultipliers().bear_volatile

    def test_dynamic_multipliers_insufficient_data(self):
        """Test returns defaults with insufficient data."""
        regime_history = [("bull_quiet", 0.05)]  # Too few

        multipliers = dynamic_regime_multipliers(regime_history)

        # Should return default multipliers
        defaults = RegimeMultipliers()
        assert multipliers.bull_quiet == defaults.bull_quiet

    def test_dynamic_multipliers_mixed_performance(self):
        """Test multipliers with mixed regime performance."""
        regime_history = [
            ("bull_quiet", 0.05),
            ("bull_quiet", -0.02),
            ("bull_quiet", 0.03),
            ("bull_quiet", 0.04),
        ]

        multipliers = dynamic_regime_multipliers(regime_history)

        # Should adjust based on overall positive performance
        assert multipliers.bull_quiet > 0

    def test_dynamic_multipliers_volatility_adjustment(self):
        """Test volatility penalty in dynamic multipliers."""
        # High volatility performance
        volatile_history = [
            ("sideways_quiet", 0.1),
            ("sideways_quiet", -0.08),
            ("sideways_quiet", 0.12),
            ("sideways_quiet", -0.09),
        ]

        # Low volatility performance
        stable_history = [
            ("sideways_quiet", 0.02),
            ("sideways_quiet", 0.03),
            ("sideways_quiet", 0.02),
            ("sideways_quiet", 0.03),
        ]

        volatile_mult = dynamic_regime_multipliers(volatile_history, volatility_adjustment=True)
        stable_mult = dynamic_regime_multipliers(stable_history, volatility_adjustment=True)

        # Stable performance should get higher multiplier due to lower volatility
        assert stable_mult.sideways_quiet >= volatile_mult.sideways_quiet

    def test_dynamic_multipliers_no_volatility_adjustment(self):
        """Test dynamic multipliers without volatility adjustment."""
        regime_history = [
            ("bull_quiet", 0.05),
            ("bull_quiet", 0.04),
            ("bull_quiet", 0.06),
        ]

        multipliers = dynamic_regime_multipliers(regime_history, volatility_adjustment=False)
        assert multipliers.bull_quiet > 0

    def test_dynamic_multipliers_bounds(self):
        """Test dynamic multipliers stay within bounds."""
        # Extremely positive performance
        extreme_history = [(f"bull_quiet", 0.5) for _ in range(10)]

        multipliers = dynamic_regime_multipliers(extreme_history)

        # Should be capped at reasonable maximum (2.0)
        assert multipliers.bull_quiet <= 2.0


class TestRegimeTransition:
    """Test regime transition adjustments."""

    def test_transition_same_regime_no_adjustment(self):
        """Test no adjustment when regime hasn't changed."""
        adjusted = regime_transition_adjustment(
            current_regime="bull_quiet",
            previous_regime="bull_quiet",
            transition_confidence=0.9,
            base_multiplier=1.2,
        )

        assert adjusted == 1.2

    def test_transition_low_confidence_conservative(self):
        """Test conservative adjustment with low transition confidence."""
        adjusted = regime_transition_adjustment(
            current_regime="bear_quiet",
            previous_regime="bull_quiet",
            transition_confidence=0.5,  # Low confidence
            base_multiplier=0.6,
        )

        assert adjusted < 0.6  # Should be more conservative

    def test_transition_high_confidence_full_multiplier(self):
        """Test full multiplier with high transition confidence."""
        adjusted = regime_transition_adjustment(
            current_regime="sideways_volatile",
            previous_regime="bull_volatile",
            transition_confidence=0.9,
            base_multiplier=0.7,
        )

        assert adjusted == 0.7

    def test_transition_moderate_confidence(self):
        """Test moderate adjustment with moderate confidence."""
        adjusted = regime_transition_adjustment(
            current_regime="crisis",
            previous_regime="bear_volatile",
            transition_confidence=0.65,
            base_multiplier=0.2,
        )

        assert 0.1 < adjusted <= 0.2


class TestRegimeMomentum:
    """Test regime momentum factor calculations."""

    def test_momentum_bull_long_duration(self):
        """Test bull regime gets momentum bonus with long duration."""
        factor = regime_momentum_factor(regime_duration_days=30, regime="bull_quiet")

        assert factor > 1.0  # Momentum bonus
        assert factor <= 1.2

    def test_momentum_bull_short_duration(self):
        """Test bull regime gets no bonus with short duration."""
        factor = regime_momentum_factor(regime_duration_days=10, regime="bull_quiet")

        assert factor == 1.0

    def test_momentum_bear_long_duration(self):
        """Test bear regime gets penalty with long duration (mean reversion)."""
        factor = regime_momentum_factor(regime_duration_days=25, regime="bear_quiet")

        assert factor < 1.0  # Mean reversion expectation
        assert factor >= 0.8

    def test_momentum_bear_short_duration(self):
        """Test bear regime gets no adjustment with short duration."""
        factor = regime_momentum_factor(regime_duration_days=10, regime="bear_volatile")

        assert factor == 1.0

    def test_momentum_crisis_no_bonus(self):
        """Test crisis regime never gets momentum bonus."""
        factor = regime_momentum_factor(regime_duration_days=30, regime="crisis")

        assert factor <= 0.8

    def test_momentum_sideways_long_duration(self):
        """Test sideways regime gets slight bonus with long duration."""
        factor = regime_momentum_factor(regime_duration_days=40, regime="sideways_quiet")

        assert factor > 1.0
        assert factor <= 1.1


class TestPortfolioRegimeAllocation:
    """Test portfolio allocation across regimes."""

    def test_portfolio_allocation_single_asset(self):
        """Test allocation with single asset."""
        regimes = {"BTC": "bull_quiet"}
        confidences = {"BTC": 0.8}

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.1)

        assert "BTC" in allocations
        assert allocations["BTC"] <= 0.1

    def test_portfolio_allocation_multiple_assets(self):
        """Test allocation across multiple assets."""
        regimes = {"BTC": "bull_quiet", "ETH": "bull_volatile", "SOL": "sideways_quiet"}
        confidences = {"BTC": 0.8, "ETH": 0.7, "SOL": 0.6}

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.15)

        # All should get some allocation
        assert all(alloc > 0 for alloc in allocations.values())

        # Total should be close to budget
        total = sum(allocations.values())
        assert abs(total - 0.15) < 0.01

        # Bull regimes should get more
        assert allocations["BTC"] > allocations["SOL"]

    def test_portfolio_allocation_crisis_regime(self):
        """Test allocation reduces for crisis regimes."""
        regimes = {"BTC": "bull_quiet", "ETH": "crisis"}
        confidences = {"BTC": 0.8, "ETH": 0.8}

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.1)

        # Crisis should get much less allocation
        assert allocations["BTC"] > allocations["ETH"]

    def test_portfolio_allocation_empty_regimes(self):
        """Test allocation with no regimes."""
        allocations = portfolio_regime_allocation({}, {}, total_risk_budget=0.1)
        assert allocations == {}

    def test_portfolio_allocation_varying_confidence(self):
        """Test higher confidence gets larger allocation."""
        regimes = {"BTC": "bull_quiet", "ETH": "bull_quiet"}
        confidences = {"BTC": 0.9, "ETH": 0.5}

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.1)

        # Higher confidence should get more
        assert allocations["BTC"] > allocations["ETH"]


class TestRegimeCorrelationAdjustment:
    """Test correlation-based regime adjustments."""

    def test_correlation_single_asset(self):
        """Test correlation adjustment with single asset."""
        regimes = {"BTC": "bull_quiet"}

        adjustments = regime_correlation_adjustment(regimes)

        assert adjustments["BTC"] == 1.0  # No adjustment

    def test_correlation_diversified_regimes(self):
        """Test no penalty with diversified regimes."""
        regimes = {"BTC": "bull_quiet", "ETH": "bear_volatile", "SOL": "sideways_quiet"}

        adjustments = regime_correlation_adjustment(regimes)

        # All should be close to 1.0 (no penalty)
        assert all(adj > 0.9 for adj in adjustments.values())

    def test_correlation_concentrated_regime(self):
        """Test penalty for concentration in one regime."""
        regimes = {
            "BTC": "bull_quiet",
            "ETH": "bull_quiet",
            "SOL": "bull_quiet",
            "AVAX": "bull_quiet",
        }

        adjustments = regime_correlation_adjustment(regimes)

        # All should be penalized (all in same regime)
        assert all(adj < 1.0 for adj in adjustments.values())

    def test_correlation_partial_concentration(self):
        """Test partial penalty for partial concentration."""
        regimes = {
            "BTC": "bull_quiet",
            "ETH": "bull_quiet",
            "SOL": "bear_volatile",
        }

        adjustments = regime_correlation_adjustment(regimes)

        # Bull assets should be penalized, bear less so
        assert adjustments["BTC"] < 1.0
        assert adjustments["ETH"] < 1.0
        assert adjustments["SOL"] == 1.0  # Unique regime


class TestRegimeVolatilityScaling:
    """Test regime-specific volatility scaling."""

    def test_volatility_scaling_crisis_regime(self):
        """Test crisis regime is very sensitive to volatility."""
        scaling = regime_volatility_scaling(
            regime="crisis", realized_volatility=0.4, expected_volatility=0.2
        )

        # 2x expected volatility should severely reduce position
        assert scaling < 0.5

    def test_volatility_scaling_volatile_regime(self):
        """Test volatile regime is less sensitive."""
        scaling = regime_volatility_scaling(
            regime="bull_volatile", realized_volatility=0.4, expected_volatility=0.3
        )

        # Only slightly above expected
        assert scaling > 0.8

    def test_volatility_scaling_quiet_regime(self):
        """Test quiet regime is very sensitive to volatility increases."""
        scaling = regime_volatility_scaling(
            regime="sideways_quiet", realized_volatility=0.3, expected_volatility=0.1
        )

        # 3x expected volatility in quiet regime
        assert scaling < 0.5

    def test_volatility_scaling_at_expected(self):
        """Test no adjustment when volatility matches expected."""
        scaling = regime_volatility_scaling(
            regime="bull_quiet", realized_volatility=0.2, expected_volatility=0.2
        )

        assert abs(scaling - 1.0) < 0.1

    def test_volatility_scaling_lower_than_expected(self):
        """Test can increase position when volatility below expected."""
        scaling = regime_volatility_scaling(
            regime="bull_quiet", realized_volatility=0.1, expected_volatility=0.2
        )

        assert scaling >= 1.0

    def test_volatility_scaling_zero_expected(self):
        """Test handles zero expected volatility."""
        scaling = regime_volatility_scaling(
            regime="bull_quiet", realized_volatility=0.2, expected_volatility=0.0
        )

        assert scaling == 1.0  # No adjustment possible


class TestSafeRegimeCalculation:
    """Test safe regime calculation with validation."""

    def test_safe_calculation_valid_regime(self):
        """Test safe calculation with valid regime."""
        result = safe_regime_calculation(regime="bull_quiet", base_multiplier=1.2, confidence=0.8)

        assert 0.1 <= result <= 3.0
        assert result > 0

    def test_safe_calculation_invalid_regime(self):
        """Test safe calculation handles invalid regime."""
        result = safe_regime_calculation(
            regime="invalid_regime", base_multiplier=1.2, confidence=0.8
        )

        # Should return neutral multiplier
        assert result == 1.0

    def test_safe_calculation_extreme_multiplier(self):
        """Test safe calculation clamps extreme multipliers."""
        result = safe_regime_calculation(regime="bull_quiet", base_multiplier=10.0, confidence=1.0)

        # Should be clamped to max 3.0
        assert result <= 3.0

    def test_safe_calculation_low_confidence(self):
        """Test safe calculation adjusts for low confidence."""
        high_conf = safe_regime_calculation(
            regime="bull_quiet", base_multiplier=1.5, confidence=0.9
        )
        low_conf = safe_regime_calculation(regime="bull_quiet", base_multiplier=1.5, confidence=0.3)

        # Low confidence should produce more conservative multiplier
        assert low_conf < high_conf

    def test_safe_calculation_zero_confidence(self):
        """Test safe calculation with zero confidence."""
        result = safe_regime_calculation(regime="bull_quiet", base_multiplier=1.5, confidence=0.0)

        # Should return conservative neutral
        assert result <= 1.0

    def test_safe_calculation_invalid_confidence(self):
        """Test safe calculation handles invalid confidence."""
        result = safe_regime_calculation(regime="bull_quiet", base_multiplier=1.2, confidence=2.0)

        # Should return conservative default
        assert result == 0.8


class TestRegimeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_regime_adjustment_negative_base_size(self):
        """Test regime adjustment rejects negative base size."""
        multipliers = RegimeMultipliers()

        with pytest.raises(ValidationError):
            regime_adjusted_size(-0.1, "bull_quiet", multipliers)

    def test_regime_adjustment_large_base_size(self):
        """Test regime adjustment handles large base sizes."""
        multipliers = RegimeMultipliers()

        adjusted, _ = regime_adjusted_size(0.9, "bull_quiet", multipliers)

        # Should be capped at 1.0
        assert adjusted <= 1.0

    def test_dynamic_multipliers_empty_history(self):
        """Test dynamic multipliers with empty history."""
        multipliers = dynamic_regime_multipliers([])

        # Should return defaults
        defaults = RegimeMultipliers()
        assert multipliers.bull_quiet == defaults.bull_quiet

    def test_momentum_zero_duration(self):
        """Test momentum with zero duration."""
        factor = regime_momentum_factor(regime_duration_days=0, regime="bull_quiet")
        assert factor == 1.0

    def test_momentum_negative_duration(self):
        """Test momentum handles negative duration."""
        factor = regime_momentum_factor(regime_duration_days=-5, regime="bull_quiet")
        assert factor == 1.0

    def test_portfolio_allocation_zero_budget(self):
        """Test portfolio allocation with zero budget."""
        regimes = {"BTC": "bull_quiet", "ETH": "bull_volatile"}
        confidences = {"BTC": 0.8, "ETH": 0.7}

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.0)

        assert all(alloc == 0 for alloc in allocations.values())

    def test_portfolio_allocation_missing_confidence(self):
        """Test portfolio allocation handles missing confidence."""
        regimes = {"BTC": "bull_quiet", "ETH": "bull_volatile"}
        confidences = {"BTC": 0.8}  # Missing ETH

        allocations = portfolio_regime_allocation(regimes, confidences, total_risk_budget=0.1)

        # Should use default confidence (0.5) for ETH
        assert "ETH" in allocations
        assert allocations["ETH"] > 0
