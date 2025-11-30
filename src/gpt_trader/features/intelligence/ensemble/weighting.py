"""
Dynamic weight calculator for ensemble strategies.

Calculates strategy weights based on:
1. Base weights (static configuration)
2. Regime adjustments (multipliers per regime)
3. Regime confidence (blend toward base when uncertain)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.features.intelligence.regime.models import RegimeType


class DynamicWeightCalculator:
    """Calculates strategy weights dynamically based on market regime.

    Weights are calculated as:
        effective_weight = base_weight * regime_multiplier * confidence_blend

    Where confidence_blend smoothly transitions from equal weights
    (low regime confidence) to full regime adjustment (high confidence).

    Example:
        calculator = DynamicWeightCalculator(
            base_weights={"baseline": 0.5, "mean_reversion": 0.5},
            regime_adjustments={
                RegimeType.BULL_QUIET.name: {"baseline": 0.8, "mean_reversion": 1.2},
            },
        )

        weights = calculator.calculate(
            regime=RegimeType.BULL_QUIET,
            confidence=0.8,
            strategy_names=["baseline", "mean_reversion"],
        )
    """

    def __init__(
        self,
        base_weights: dict[str, float],
        regime_adjustments: dict[str, dict[str, float]],
    ):
        """Initialize calculator.

        Args:
            base_weights: Default weights per strategy (should sum to ~1.0)
            regime_adjustments: Per-regime weight multipliers
                Keys are RegimeType names (e.g., "BULL_QUIET")
                Values are dicts mapping strategy name to multiplier
        """
        self.base_weights = base_weights
        self.regime_adjustments = regime_adjustments

    def calculate(
        self,
        regime: RegimeType,
        confidence: float,
        strategy_names: list[str],
    ) -> dict[str, float]:
        """Calculate normalized weights for current regime.

        Args:
            regime: Current market regime
            confidence: Regime detection confidence (0.0 to 1.0)
            strategy_names: List of strategy names to calculate weights for

        Returns:
            Dictionary mapping strategy name to normalized weight (sums to 1.0)
        """
        weights: dict[str, float] = {}

        for name in strategy_names:
            # Get base weight (default to equal distribution)
            base = self.base_weights.get(name, 1.0 / max(len(strategy_names), 1))

            # Get regime-specific multiplier
            regime_key = regime.name
            adjustments = self.regime_adjustments.get(regime_key, {})
            multiplier = adjustments.get(name, 1.0)

            # Apply confidence-weighted blending
            # High confidence = full regime adjustment
            # Low confidence = blend toward equal weights
            effective_multiplier = 1.0 + (multiplier - 1.0) * confidence

            weights[name] = base * effective_multiplier

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            equal = 1.0 / max(len(strategy_names), 1)
            weights = {name: equal for name in strategy_names}

        return weights

    def get_regime_bias(self, regime: RegimeType) -> dict[str, str]:
        """Get human-readable description of regime bias.

        Args:
            regime: Market regime

        Returns:
            Dictionary mapping strategy name to bias description
        """
        regime_key = regime.name
        adjustments = self.regime_adjustments.get(regime_key, {})

        bias: dict[str, str] = {}
        for name, multiplier in adjustments.items():
            if multiplier > 1.2:
                bias[name] = "strongly_favored"
            elif multiplier > 1.0:
                bias[name] = "slightly_favored"
            elif multiplier < 0.5:
                bias[name] = "strongly_disfavored"
            elif multiplier < 1.0:
                bias[name] = "slightly_disfavored"
            else:
                bias[name] = "neutral"

        return bias


__all__ = ["DynamicWeightCalculator"]
