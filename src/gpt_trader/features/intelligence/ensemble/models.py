"""
Ensemble data models.

Defines the core types for multi-signal ensemble:
- StrategyVote: Individual strategy decision with weight
- EnsembleConfig: Configuration for ensemble orchestration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gpt_trader.features.intelligence.regime.models import RegimeType

if TYPE_CHECKING:
    from gpt_trader.core import Decision


@dataclass
class StrategyVote:
    """A strategy's vote in the ensemble.

    Attributes:
        strategy_name: Name identifier for the strategy
        decision: The strategy's trading decision
        weight: Weight assigned to this strategy's vote (0.0 to 1.0)
        weighted_confidence: weight * decision.confidence
    """

    strategy_name: str
    decision: Decision
    weight: float

    @property
    def weighted_confidence(self) -> float:
        """Calculate weight-adjusted confidence."""
        return self.weight * self.decision.confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "strategy_name": self.strategy_name,
            "action": self.decision.action.value,
            "confidence": round(self.decision.confidence, 4),
            "weight": round(self.weight, 4),
            "weighted_confidence": round(self.weighted_confidence, 4),
            "reason": self.decision.reason,
        }


@dataclass
class EnsembleConfig:
    """Configuration for ensemble orchestration.

    All parameters are tunable via YAML profiles.
    """

    # Voting mechanism
    voting_method: str = "confidence_leader"  # or "weighted_majority"

    # Base strategy weights (before regime adjustment)
    base_weights: dict[str, float] = field(
        default_factory=lambda: {
            "baseline": 0.4,
            "mean_reversion": 0.4,
        }
    )

    # Regime-specific weight adjustments (multipliers)
    # Values > 1.0 increase weight, < 1.0 decrease weight
    regime_weight_adjustments: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            RegimeType.BULL_QUIET.name: {
                "baseline": 0.8,
                "mean_reversion": 1.5,
            },
            RegimeType.BULL_VOLATILE.name: {
                "baseline": 1.2,
                "mean_reversion": 0.6,
            },
            RegimeType.BEAR_QUIET.name: {
                "baseline": 0.6,
                "mean_reversion": 1.3,
            },
            RegimeType.BEAR_VOLATILE.name: {
                "baseline": 0.4,
                "mean_reversion": 0.4,
            },
            RegimeType.SIDEWAYS_QUIET.name: {
                "baseline": 0.5,
                "mean_reversion": 1.8,
            },
            RegimeType.SIDEWAYS_VOLATILE.name: {
                "baseline": 0.7,
                "mean_reversion": 1.0,
            },
            RegimeType.CRISIS.name: {
                "baseline": 0.2,
                "mean_reversion": 0.2,
            },
        }
    )

    # Confidence requirements
    min_ensemble_confidence: float = 0.4  # Below this, output HOLD

    # Crisis behavior
    crisis_behavior: str = "scaled_down"  # or "reduce_only", "halt"
    crisis_scale_factor: float = 0.2  # Scale position sizes to 20%

    # Adaptive learning
    enable_adaptive_learning: bool = False  # Enable Bayesian weight adaptation
    adaptive_smoothing: float = 0.7  # Weight smoothing (0 = instant, 1 = no update)
    adaptive_min_weight: float = 0.05  # Minimum weight floor
    adaptive_max_weight: float = 0.8  # Maximum weight cap

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "voting_method": self.voting_method,
            "base_weights": self.base_weights,
            "regime_weight_adjustments": self.regime_weight_adjustments,
            "min_ensemble_confidence": self.min_ensemble_confidence,
            "crisis_behavior": self.crisis_behavior,
            "crisis_scale_factor": self.crisis_scale_factor,
            "enable_adaptive_learning": self.enable_adaptive_learning,
            "adaptive_smoothing": self.adaptive_smoothing,
            "adaptive_min_weight": self.adaptive_min_weight,
            "adaptive_max_weight": self.adaptive_max_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleConfig:
        """Create config from dictionary."""
        return cls(
            voting_method=data.get("voting_method", "confidence_leader"),
            base_weights=data.get("base_weights", {"baseline": 0.4, "mean_reversion": 0.4}),
            regime_weight_adjustments=data.get("regime_weight_adjustments", {}),
            min_ensemble_confidence=data.get("min_ensemble_confidence", 0.4),
            crisis_behavior=data.get("crisis_behavior", "scaled_down"),
            crisis_scale_factor=data.get("crisis_scale_factor", 0.2),
            enable_adaptive_learning=data.get("enable_adaptive_learning", False),
            adaptive_smoothing=data.get("adaptive_smoothing", 0.7),
            adaptive_min_weight=data.get("adaptive_min_weight", 0.05),
            adaptive_max_weight=data.get("adaptive_max_weight", 0.8),
        )


__all__ = [
    "EnsembleConfig",
    "StrategyVote",
]
