"""Tests for DynamicWeightCalculator."""

import pytest

from gpt_trader.features.intelligence.ensemble.weighting import DynamicWeightCalculator
from gpt_trader.features.intelligence.regime.models import RegimeType


class TestDynamicWeightCalculator:
    """Test DynamicWeightCalculator."""

    @pytest.fixture
    def calculator(self) -> DynamicWeightCalculator:
        """Create calculator with test config."""
        return DynamicWeightCalculator(
            base_weights={"baseline": 0.5, "mean_reversion": 0.5},
            regime_adjustments={
                RegimeType.BULL_QUIET.name: {
                    "baseline": 0.8,
                    "mean_reversion": 1.2,
                },
                RegimeType.SIDEWAYS_QUIET.name: {
                    "baseline": 0.5,
                    "mean_reversion": 1.5,
                },
                RegimeType.CRISIS.name: {
                    "baseline": 0.2,
                    "mean_reversion": 0.2,
                },
            },
        )

    def test_equal_weights_with_no_adjustment(self, calculator: DynamicWeightCalculator):
        """Test equal weights when no regime adjustment exists."""
        weights = calculator.calculate(
            regime=RegimeType.UNKNOWN,
            confidence=1.0,
            strategy_names=["baseline", "mean_reversion"],
        )

        # Should be normalized to sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.001
        # Equal base weights, no adjustment
        assert abs(weights["baseline"] - 0.5) < 0.001
        assert abs(weights["mean_reversion"] - 0.5) < 0.001

    def test_regime_adjustment_applied(self, calculator: DynamicWeightCalculator):
        """Test that regime adjustments are applied."""
        weights = calculator.calculate(
            regime=RegimeType.SIDEWAYS_QUIET,
            confidence=1.0,
            strategy_names=["baseline", "mean_reversion"],
        )

        # Mean reversion should be favored (1.5x) vs baseline (0.5x)
        assert weights["mean_reversion"] > weights["baseline"]
        # Should still sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_confidence_blending(self, calculator: DynamicWeightCalculator):
        """Test confidence-based blending toward base weights."""
        high_conf_weights = calculator.calculate(
            regime=RegimeType.SIDEWAYS_QUIET,
            confidence=1.0,  # Full regime adjustment
            strategy_names=["baseline", "mean_reversion"],
        )

        low_conf_weights = calculator.calculate(
            regime=RegimeType.SIDEWAYS_QUIET,
            confidence=0.0,  # No regime adjustment, equal weights
            strategy_names=["baseline", "mean_reversion"],
        )

        # Low confidence should be closer to equal weights
        high_diff = abs(high_conf_weights["mean_reversion"] - high_conf_weights["baseline"])
        low_diff = abs(low_conf_weights["mean_reversion"] - low_conf_weights["baseline"])

        # Higher confidence = more extreme difference
        assert high_diff > low_diff

    def test_crisis_reduces_all_weights(self, calculator: DynamicWeightCalculator):
        """Test that crisis regime reduces all strategy weights proportionally."""
        crisis_weights = calculator.calculate(
            regime=RegimeType.CRISIS,
            confidence=1.0,
            strategy_names=["baseline", "mean_reversion"],
        )

        calculator.calculate(
            regime=RegimeType.BULL_QUIET,
            confidence=1.0,
            strategy_names=["baseline", "mean_reversion"],
        )

        # In crisis, both multipliers are 0.2, so should be roughly equal
        assert abs(crisis_weights["baseline"] - crisis_weights["mean_reversion"]) < 0.1
        # But they should still sum to 1.0
        assert abs(sum(crisis_weights.values()) - 1.0) < 0.001

    def test_handles_unknown_strategy(self, calculator: DynamicWeightCalculator):
        """Test handling of strategy not in base_weights."""
        weights = calculator.calculate(
            regime=RegimeType.BULL_QUIET,
            confidence=1.0,
            strategy_names=["baseline", "mean_reversion", "new_strategy"],
        )

        # Should include all strategies
        assert "new_strategy" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_get_regime_bias(self, calculator: DynamicWeightCalculator):
        """Test human-readable regime bias."""
        bias = calculator.get_regime_bias(RegimeType.SIDEWAYS_QUIET)

        # 0.5x is slightly_disfavored (< 1.0 but >= 0.5)
        assert bias.get("baseline") == "slightly_disfavored"
        assert bias.get("mean_reversion") == "strongly_favored"  # 1.5x > 1.2

    def test_normalization_with_extreme_multipliers(self):
        """Test that weights are properly normalized with extreme multipliers."""
        calculator = DynamicWeightCalculator(
            base_weights={"a": 0.5, "b": 0.5},
            regime_adjustments={
                RegimeType.CRISIS.name: {
                    "a": 0.01,  # Very small
                    "b": 10.0,  # Very large
                },
            },
        )

        weights = calculator.calculate(
            regime=RegimeType.CRISIS,
            confidence=1.0,
            strategy_names=["a", "b"],
        )

        # Should still sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.001
        # b should dominate
        assert weights["b"] > 0.9
