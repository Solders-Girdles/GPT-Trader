"""
Comprehensive tests for confidence-based position sizing.

Tests cover:
- Confidence adjustments with different curves
- Multi-model confidence aggregation
- Confidence decay over time
- Backtest metrics to confidence conversion
- Adaptive thresholds
- Position limits based on confidence
- Risk budget allocation
- Input validation
"""

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.position_sizing.confidence import (
    adaptive_confidence_threshold,
    confidence_adjusted_size,
    confidence_decay,
    confidence_from_backtest_metrics,
    confidence_position_limits,
    confidence_risk_budget,
    multi_model_confidence,
    safe_confidence_calculation,
    validate_confidence_inputs,
)
from bot_v2.features.position_sizing.types import ConfidenceAdjustment


class TestConfidenceAdjustment:
    """Test basic confidence adjustment functionality."""

    def test_confidence_adjustment_high_confidence(self):
        """Test adjustment with high confidence increases position."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=0.9)

        adjusted, explanation = confidence_adjusted_size(base_size, 0.9, adjustment)

        assert adjusted > base_size
        assert "0.90" in explanation or "90" in explanation

    def test_confidence_adjustment_low_confidence(self):
        """Test adjustment with low confidence decreases position."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=0.4, min_confidence=0.6)

        adjusted, explanation = confidence_adjusted_size(base_size, 0.4, adjustment)

        assert adjusted < base_size
        assert "Low confidence" in explanation or "low confidence" in explanation.lower()

    def test_confidence_adjustment_at_threshold(self):
        """Test adjustment right at confidence threshold."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=0.6, min_confidence=0.6)

        adjusted, _ = confidence_adjusted_size(base_size, 0.6, adjustment)

        # At threshold, implementation is conservative (0.5x multiplier)
        assert adjusted == 0.05

    def test_confidence_adjustment_linear_curve(self):
        """Test linear confidence adjustment curve."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(
            confidence=0.8, min_confidence=0.6, max_adjustment=2.0, adjustment_curve="linear"
        )

        adjusted, explanation = confidence_adjusted_size(base_size, 0.8, adjustment)

        assert adjusted > base_size
        assert adjusted <= base_size * 2.0
        assert "linear" in explanation

    def test_confidence_adjustment_exponential_curve(self):
        """Test exponential confidence adjustment curve."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(
            confidence=0.8, min_confidence=0.6, max_adjustment=2.0, adjustment_curve="exponential"
        )

        adjusted, explanation = confidence_adjusted_size(base_size, 0.8, adjustment)

        assert adjusted > base_size
        assert "exponential" in explanation

    def test_confidence_adjustment_sigmoid_curve(self):
        """Test sigmoid confidence adjustment curve."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(
            confidence=0.85, min_confidence=0.6, max_adjustment=1.8, adjustment_curve="sigmoid"
        )

        adjusted, explanation = confidence_adjusted_size(base_size, 0.85, adjustment)

        assert adjusted > base_size
        assert "sigmoid" in explanation

    def test_confidence_adjustment_caps_at_max(self):
        """Test that adjustment doesn't exceed max_adjustment."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=0.99, min_confidence=0.6, max_adjustment=1.5)

        adjusted, _ = confidence_adjusted_size(base_size, 0.99, adjustment)

        assert adjusted <= base_size * 1.5

    def test_confidence_adjustment_zero_confidence(self):
        """Test adjustment with zero confidence."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=0.0, min_confidence=0.6)

        adjusted, explanation = confidence_adjusted_size(base_size, 0.0, adjustment)

        assert adjusted < base_size
        assert adjusted >= 0

    def test_confidence_adjustment_perfect_confidence(self):
        """Test adjustment with perfect confidence."""
        base_size = 0.1
        adjustment = ConfidenceAdjustment(confidence=1.0, max_adjustment=2.0)

        adjusted, _ = confidence_adjusted_size(base_size, 1.0, adjustment)

        assert adjusted > base_size
        assert adjusted <= base_size * 2.0


class TestConfidenceValidation:
    """Test confidence input validation."""

    def test_validate_confidence_valid_inputs(self):
        """Test validation passes for valid inputs."""
        adjustment = ConfidenceAdjustment(
            confidence=0.8, min_confidence=0.6, max_adjustment=2.0, adjustment_curve="linear"
        )

        errors = validate_confidence_inputs(0.8, adjustment)
        assert len(errors) == 0

    def test_validate_confidence_invalid_confidence(self):
        """Test validation catches invalid confidence score."""
        adjustment = ConfidenceAdjustment(confidence=0.8)

        errors = validate_confidence_inputs(1.5, adjustment)  # > 1.0
        assert len(errors) > 0

    def test_validate_confidence_invalid_min_confidence(self):
        """Test validation catches invalid min_confidence."""
        adjustment = ConfidenceAdjustment(confidence=0.8, min_confidence=1.5)

        errors = validate_confidence_inputs(0.8, adjustment)
        assert len(errors) > 0

    def test_validate_confidence_invalid_curve(self):
        """Test validation catches invalid adjustment curve."""
        adjustment = ConfidenceAdjustment(confidence=0.8, adjustment_curve="invalid")

        errors = validate_confidence_inputs(0.8, adjustment)
        assert len(errors) > 0
        assert any("curve" in err.lower() for err in errors)

    def test_validate_confidence_negative_max_adjustment(self):
        """Test validation catches negative max_adjustment."""
        adjustment = ConfidenceAdjustment(confidence=0.8, max_adjustment=-1.0)

        errors = validate_confidence_inputs(0.8, adjustment)
        assert len(errors) > 0

    def test_confidence_adjustment_invalid_confidence_returns_zero(self):
        """Test that invalid confidence returns zero with explanation."""
        adjustment = ConfidenceAdjustment(confidence=0.8)

        adjusted, explanation = confidence_adjusted_size(0.1, 2.0, adjustment)  # Invalid

        assert adjusted == 0.0
        assert "validation" in explanation.lower() or "error" in explanation.lower()


class TestMultiModelConfidence:
    """Test multi-model confidence aggregation."""

    def test_multi_model_equal_weights(self):
        """Test multi-model with equal weights."""
        confidences = [0.7, 0.8, 0.9]

        combined = multi_model_confidence(confidences)

        # Should be average: (0.7 + 0.8 + 0.9) / 3 = 0.8
        assert abs(combined - 0.8) < 0.001

    def test_multi_model_custom_weights(self):
        """Test multi-model with custom weights."""
        confidences = [0.6, 0.8, 1.0]
        weights = [1.0, 2.0, 1.0]  # Middle model has double weight

        combined = multi_model_confidence(confidences, weights)

        # (0.6*1 + 0.8*2 + 1.0*1) / 4 = 3.2/4 = 0.8
        assert abs(combined - 0.8) < 0.001

    def test_multi_model_empty_list(self):
        """Test multi-model with empty confidence list."""
        combined = multi_model_confidence([])
        assert combined == 0.0

    def test_multi_model_single_confidence(self):
        """Test multi-model with single confidence."""
        combined = multi_model_confidence([0.75])
        assert combined == 0.75

    def test_multi_model_weight_mismatch(self):
        """Test multi-model handles weight count mismatch."""
        confidences = [0.7, 0.8, 0.9]
        weights = [1.0, 2.0]  # Wrong number of weights

        combined = multi_model_confidence(confidences, weights)

        # Should fall back to equal weights
        assert abs(combined - 0.8) < 0.001

    def test_multi_model_zero_weights(self):
        """Test multi-model with all zero weights."""
        confidences = [0.7, 0.8, 0.9]
        weights = [0.0, 0.0, 0.0]

        combined = multi_model_confidence(confidences, weights)
        assert combined == 0.0


class TestConfidenceDecay:
    """Test confidence decay over time."""

    def test_confidence_decay_no_time(self):
        """Test no decay at time zero."""
        decayed = confidence_decay(0.9, time_since_prediction=0.0, half_life_hours=24.0)
        assert decayed == 0.9

    def test_confidence_decay_one_half_life(self):
        """Test decay after one half-life."""
        decayed = confidence_decay(0.8, time_since_prediction=24.0, half_life_hours=24.0)

        # Should be half: 0.8 * 0.5 = 0.4
        assert abs(decayed - 0.4) < 0.001

    def test_confidence_decay_two_half_lives(self):
        """Test decay after two half-lives."""
        decayed = confidence_decay(0.8, time_since_prediction=48.0, half_life_hours=24.0)

        # Should be quarter: 0.8 * 0.25 = 0.2
        assert abs(decayed - 0.2) < 0.001

    def test_confidence_decay_short_half_life(self):
        """Test decay with short half-life."""
        decayed = confidence_decay(0.9, time_since_prediction=12.0, half_life_hours=6.0)

        # 12 hours = 2 half-lives, so 0.9 * 0.25 = 0.225
        assert abs(decayed - 0.225) < 0.001

    def test_confidence_decay_negative_time(self):
        """Test no decay with negative time."""
        decayed = confidence_decay(0.9, time_since_prediction=-5.0, half_life_hours=24.0)
        assert decayed == 0.9


class TestConfidenceFromBacktest:
    """Test generating confidence from backtest metrics."""

    def test_backtest_excellent_metrics(self):
        """Test confidence from excellent backtest results."""
        confidence = confidence_from_backtest_metrics(
            sharpe_ratio=3.0, win_rate=0.7, profit_factor=2.5, max_drawdown=0.05
        )

        assert confidence > 0.7
        assert confidence <= 1.0

    def test_backtest_poor_metrics(self):
        """Test confidence from poor backtest results."""
        confidence = confidence_from_backtest_metrics(
            sharpe_ratio=0.5, win_rate=0.4, profit_factor=0.9, max_drawdown=0.3
        )

        assert confidence < 0.5

    def test_backtest_mixed_metrics(self):
        """Test confidence from mixed backtest results."""
        confidence = confidence_from_backtest_metrics(
            sharpe_ratio=1.5, win_rate=0.55, profit_factor=1.3, max_drawdown=0.15
        )

        assert 0.3 < confidence < 0.8

    def test_backtest_zero_metrics(self):
        """Test confidence from zero metrics."""
        confidence = confidence_from_backtest_metrics(
            sharpe_ratio=0.0, win_rate=0.0, profit_factor=0.0, max_drawdown=0.0
        )

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_backtest_high_drawdown_penalty(self):
        """Test that high drawdown significantly penalizes confidence."""
        high_drawdown = confidence_from_backtest_metrics(
            sharpe_ratio=2.0, win_rate=0.65, profit_factor=2.0, max_drawdown=0.5  # 50% DD
        )

        low_drawdown = confidence_from_backtest_metrics(
            sharpe_ratio=2.0, win_rate=0.65, profit_factor=2.0, max_drawdown=0.05  # 5% DD
        )

        assert low_drawdown > high_drawdown


class TestAdaptiveThreshold:
    """Test adaptive confidence threshold."""

    def test_adaptive_threshold_good_performance(self):
        """Test threshold lowers with good recent performance."""
        good_returns = [0.05, 0.03, 0.04, 0.06, 0.02]

        threshold = adaptive_confidence_threshold(good_returns, base_threshold=0.6)

        # Good performance should lower threshold (more trades)
        assert threshold < 0.6

    def test_adaptive_threshold_poor_performance(self):
        """Test threshold raises with poor recent performance."""
        poor_returns = [-0.03, -0.02, -0.04, -0.01, -0.03]

        threshold = adaptive_confidence_threshold(poor_returns, base_threshold=0.6)

        # Poor performance should raise threshold (fewer trades)
        assert threshold > 0.6

    def test_adaptive_threshold_insufficient_data(self):
        """Test returns base threshold with insufficient data."""
        few_returns = [0.01, -0.01, 0.02]

        threshold = adaptive_confidence_threshold(few_returns, base_threshold=0.6)
        assert threshold == 0.6

    def test_adaptive_threshold_bounds(self):
        """Test threshold stays within reasonable bounds."""
        extreme_returns = [0.5, 0.6, 0.7, 0.8, 0.9]  # Unrealistic extreme gains

        threshold = adaptive_confidence_threshold(extreme_returns, base_threshold=0.6)

        # Should stay within 0.3 to 0.9
        assert 0.3 <= threshold <= 0.9

    def test_adaptive_threshold_mixed_performance(self):
        """Test threshold with mixed performance."""
        mixed_returns = [0.03, -0.02, 0.01, -0.01, 0.02]

        threshold = adaptive_confidence_threshold(mixed_returns, base_threshold=0.6)

        # Should be close to base threshold
        assert 0.5 <= threshold <= 0.7


class TestConfidencePositionLimits:
    """Test position limits based on confidence."""

    def test_position_limits_high_confidence(self):
        """Test position limits with high confidence allow larger positions."""
        min_pos, max_pos = confidence_position_limits(confidence=0.85, base_max_position=0.2)

        assert max_pos > 0.2  # Can go above base
        assert min_pos > 0

    def test_position_limits_moderate_confidence(self):
        """Test position limits with moderate confidence."""
        min_pos, max_pos = confidence_position_limits(confidence=0.65, base_max_position=0.2)

        assert max_pos <= 0.2  # Conservative max
        assert min_pos > 0

    def test_position_limits_low_confidence(self):
        """Test position limits with low confidence severely restrict size."""
        min_pos, max_pos = confidence_position_limits(confidence=0.3, base_max_position=0.2)

        assert max_pos < 0.1  # Much lower than base
        assert min_pos < max_pos

    def test_position_limits_very_low_confidence(self):
        """Test position limits with very low confidence."""
        min_pos, max_pos = confidence_position_limits(confidence=0.1, base_max_position=0.2)

        assert max_pos < 0.05  # Very small max
        assert min_pos > 0


class TestConfidenceRiskBudget:
    """Test risk budget allocation based on confidence."""

    def test_risk_budget_high_confidence(self):
        """Test high confidence gets larger risk budget."""
        budget = confidence_risk_budget(confidence=0.85, total_risk_budget=0.1)

        assert budget > 0.05  # More than half
        assert budget <= 0.1

    def test_risk_budget_moderate_confidence(self):
        """Test moderate confidence gets proportional budget."""
        budget = confidence_risk_budget(confidence=0.6, total_risk_budget=0.1)

        assert 0.02 < budget < 0.08

    def test_risk_budget_low_confidence(self):
        """Test low confidence gets minimal budget."""
        budget = confidence_risk_budget(confidence=0.3, total_risk_budget=0.1)

        assert budget < 0.01

    def test_risk_budget_zero_confidence(self):
        """Test zero confidence gets minimal budget."""
        budget = confidence_risk_budget(confidence=0.0, total_risk_budget=0.1)

        assert budget >= 0
        assert budget < 0.005


class TestSafeConfidenceCalculation:
    """Test safe confidence calculation with validation."""

    def test_safe_calculation_valid_inputs(self):
        """Test safe calculation with valid inputs."""
        confidences = [0.7, 0.8, 0.9]

        result = safe_confidence_calculation(confidences)

        assert 0.7 <= result <= 0.9

    def test_safe_calculation_empty_list(self):
        """Test safe calculation with empty list."""
        result = safe_confidence_calculation([])
        assert result == 0.0

    def test_safe_calculation_invalid_confidence(self):
        """Test safe calculation handles invalid confidence gracefully."""
        confidences = [0.7, 1.5, 0.9]  # Invalid

        result = safe_confidence_calculation(confidences)

        # Should return conservative default
        assert result == 0.5

    def test_safe_calculation_with_weights(self):
        """Test safe calculation with valid weights."""
        confidences = [0.6, 0.8, 1.0]
        weights = [1.0, 2.0, 1.0]

        result = safe_confidence_calculation(confidences, weights)

        assert 0.6 <= result <= 1.0

    def test_safe_calculation_weight_mismatch(self):
        """Test safe calculation handles weight mismatch."""
        confidences = [0.7, 0.8, 0.9]
        weights = [1.0, 2.0]  # Wrong count

        result = safe_confidence_calculation(confidences, weights)

        # Should fall back to equal weights
        assert 0.7 <= result <= 0.9

    def test_safe_calculation_negative_weights(self):
        """Test safe calculation rejects negative weights."""
        confidences = [0.7, 0.8, 0.9]
        weights = [1.0, -1.0, 1.0]  # Invalid

        result = safe_confidence_calculation(confidences, weights)

        # Should return conservative default
        assert result == 0.5


class TestConfidenceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_confidence_adjustment_with_zero_base_size(self):
        """Test adjustment with zero base size."""
        adjustment = ConfidenceAdjustment(confidence=0.8)

        adjusted, _ = confidence_adjusted_size(0.0, 0.8, adjustment)
        assert adjusted == 0.0

    def test_confidence_adjustment_very_small_base_size(self):
        """Test adjustment with very small base size."""
        adjustment = ConfidenceAdjustment(confidence=0.9, max_adjustment=2.0)

        adjusted, _ = confidence_adjusted_size(0.001, 0.9, adjustment)

        assert adjusted > 0.001
        assert adjusted <= 0.002

    def test_multi_model_all_zeros(self):
        """Test multi-model with all zero confidences."""
        combined = multi_model_confidence([0.0, 0.0, 0.0])
        assert combined == 0.0

    def test_multi_model_all_ones(self):
        """Test multi-model with all perfect confidences."""
        combined = multi_model_confidence([1.0, 1.0, 1.0])
        assert combined == 1.0

    def test_confidence_decay_zero_half_life(self):
        """Test decay with zero half-life."""
        # Should handle gracefully without divide by zero
        decayed = confidence_decay(0.9, time_since_prediction=1.0, half_life_hours=0.0001)
        assert 0 <= decayed <= 0.9

    def test_backtest_negative_sharpe(self):
        """Test backtest confidence with negative Sharpe ratio."""
        confidence = confidence_from_backtest_metrics(
            sharpe_ratio=-1.0, win_rate=0.3, profit_factor=0.5, max_drawdown=0.2
        )

        assert confidence >= 0.0  # Should not go negative

    def test_position_limits_boundary_confidences(self):
        """Test position limits at boundary confidences."""
        # At 0.5 (boundary)
        min_pos, max_pos = confidence_position_limits(confidence=0.5, base_max_position=0.2)
        assert min_pos >= 0
        assert max_pos > min_pos

        # At 0.7 (boundary)
        min_pos, max_pos = confidence_position_limits(confidence=0.7, base_max_position=0.2)
        assert min_pos >= 0
        assert max_pos > min_pos

    def test_risk_budget_boundary_confidences(self):
        """Test risk budget at boundary confidences."""
        # At 0.5 (boundary)
        budget_50 = confidence_risk_budget(confidence=0.5, total_risk_budget=0.1)
        assert budget_50 >= 0

        # At 0.7 (boundary)
        budget_70 = confidence_risk_budget(confidence=0.7, total_risk_budget=0.1)
        assert budget_70 >= budget_50  # Higher confidence gets more
