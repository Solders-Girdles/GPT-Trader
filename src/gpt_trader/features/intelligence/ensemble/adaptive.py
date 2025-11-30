"""
Bayesian adaptive learning for ensemble strategy weights.

Uses Beta distributions to model strategy success probabilities per regime,
enabling adaptive weight adjustment based on observed performance.

Key concepts:
- Each strategy has a Beta(alpha, beta) prior for each regime
- Alpha counts "successes", beta counts "failures"
- Posterior updates via conjugate prior: Beta(alpha + wins, beta + losses)
- Expected success rate: alpha / (alpha + beta)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..regime.models import RegimeType


@dataclass
class StrategyPerformanceRecord:
    """Tracks performance metrics for a single strategy.

    Maintains per-regime success/failure counts for Bayesian updates.
    """

    strategy_name: str

    # Per-regime performance: regime_name -> (alpha, beta) Beta distribution params
    # Alpha = successes + prior, Beta = failures + prior
    # Start with Beta(2, 2) prior (weakly informative, centered at 0.5)
    _regime_performance: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Overall performance (across all regimes)
    total_trades: int = 0
    total_wins: int = 0
    total_pnl: float = 0.0

    # Recent performance window
    _recent_outcomes: list[bool] = field(default_factory=list)
    recent_window_size: int = 50

    def __post_init__(self) -> None:
        """Initialize with weakly informative priors."""
        # Initialize all known regimes with Beta(2, 2) prior
        for regime in RegimeType:
            if regime.name not in self._regime_performance:
                self._regime_performance[regime.name] = (2.0, 2.0)

    def record_outcome(
        self,
        regime: RegimeType,
        is_success: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record a trade outcome for Bayesian update.

        Args:
            regime: Market regime when trade occurred
            is_success: Whether trade was profitable (True) or not (False)
            pnl: Profit/loss amount (optional, for tracking)
        """
        # Update regime-specific beta distribution
        alpha, beta = self._regime_performance.get(regime.name, (2.0, 2.0))
        if is_success:
            alpha += 1.0
        else:
            beta += 1.0
        self._regime_performance[regime.name] = (alpha, beta)

        # Update totals
        self.total_trades += 1
        if is_success:
            self.total_wins += 1
        self.total_pnl += pnl

        # Update recent window
        self._recent_outcomes.append(is_success)
        if len(self._recent_outcomes) > self.recent_window_size:
            self._recent_outcomes.pop(0)

    def get_success_probability(self, regime: RegimeType) -> float:
        """Get expected success probability for regime.

        Uses Beta distribution mean: alpha / (alpha + beta)

        Args:
            regime: Market regime

        Returns:
            Expected success probability (0.0 to 1.0)
        """
        alpha, beta = self._regime_performance.get(regime.name, (2.0, 2.0))
        return alpha / (alpha + beta)

    def get_confidence_interval(
        self, regime: RegimeType, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Get confidence interval for success probability.

        Uses normal approximation to Beta distribution for CI.

        Args:
            regime: Market regime
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha, beta = self._regime_performance.get(regime.name, (2.0, 2.0))
        n = alpha + beta

        # Mean and variance of Beta distribution
        mean = alpha / n
        variance = (alpha * beta) / (n * n * (n + 1))
        std_dev = math.sqrt(variance)

        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        lower = max(0.0, mean - z * std_dev)
        upper = min(1.0, mean + z * std_dev)

        return (lower, upper)

    def get_uncertainty(self, regime: RegimeType) -> float:
        """Get uncertainty level for regime (higher = less data).

        Uses Beta distribution variance as uncertainty measure.

        Args:
            regime: Market regime

        Returns:
            Uncertainty score (0.0 to 1.0, lower = more confident)
        """
        alpha, beta = self._regime_performance.get(regime.name, (2.0, 2.0))
        n = alpha + beta

        # Variance of Beta distribution
        variance = (alpha * beta) / (n * n * (n + 1))

        # Max variance is 0.25 at Beta(1, 1), normalize to 0-1
        return min(1.0, variance * 4)

    def get_recent_win_rate(self) -> float:
        """Get win rate from recent trades.

        Returns:
            Recent win rate (0.0 to 1.0)
        """
        if not self._recent_outcomes:
            return 0.5
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    def serialize(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "strategy_name": self.strategy_name,
            "regime_performance": self._regime_performance,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_pnl": self.total_pnl,
            "recent_outcomes": self._recent_outcomes,
            "recent_window_size": self.recent_window_size,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> StrategyPerformanceRecord:
        """Restore from serialized data."""
        record = cls(strategy_name=data["strategy_name"])
        record._regime_performance = data.get("regime_performance", {})
        record.total_trades = data.get("total_trades", 0)
        record.total_wins = data.get("total_wins", 0)
        record.total_pnl = data.get("total_pnl", 0.0)
        record._recent_outcomes = data.get("recent_outcomes", [])
        record.recent_window_size = data.get("recent_window_size", 50)
        return record


@dataclass
class BayesianWeightConfig:
    """Configuration for Bayesian weight adaptation."""

    # Prior strength (higher = slower adaptation)
    prior_strength: float = 2.0

    # Minimum weight floor (prevents complete exclusion)
    min_weight: float = 0.05

    # Maximum weight cap (prevents over-concentration)
    max_weight: float = 0.8

    # Weight smoothing factor (0 = instant update, 1 = no update)
    smoothing: float = 0.7

    # Uncertainty penalty (reduce weight for uncertain strategies)
    uncertainty_penalty: float = 0.3

    # Recent performance bonus weight
    recent_bonus_weight: float = 0.2


class BayesianWeightUpdater:
    """Updates ensemble weights based on Bayesian inference.

    Maintains Beta distribution beliefs about each strategy's success
    probability in each regime, and uses these to compute adaptive weights.

    Example:
        updater = BayesianWeightUpdater(strategy_names=["baseline", "mean_reversion"])

        # Record trade outcomes
        updater.record_outcome(
            strategy_name="baseline",
            regime=RegimeType.BULL_QUIET,
            is_success=True,
            pnl=100.0,
        )

        # Get updated weights for current regime
        weights = updater.get_weights(RegimeType.BULL_QUIET)
    """

    def __init__(
        self,
        strategy_names: list[str],
        base_weights: dict[str, float] | None = None,
        config: BayesianWeightConfig | None = None,
    ):
        """Initialize Bayesian weight updater.

        Args:
            strategy_names: List of strategy names to track
            base_weights: Initial base weights (default: equal)
            config: Configuration for weight updates
        """
        self.strategy_names = strategy_names
        self.config = config or BayesianWeightConfig()

        # Initialize base weights
        if base_weights is None:
            equal_weight = 1.0 / len(strategy_names) if strategy_names else 0.5
            self.base_weights = {name: equal_weight for name in strategy_names}
        else:
            self.base_weights = base_weights.copy()

        # Initialize performance records
        self._performance: dict[str, StrategyPerformanceRecord] = {
            name: StrategyPerformanceRecord(strategy_name=name) for name in strategy_names
        }

        # Current adapted weights (per regime)
        self._current_weights: dict[str, dict[str, float]] = {}

    def record_outcome(
        self,
        strategy_name: str,
        regime: RegimeType,
        is_success: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record a trade outcome for a strategy.

        Args:
            strategy_name: Name of strategy that made the decision
            regime: Market regime when trade occurred
            is_success: Whether trade was profitable
            pnl: Profit/loss amount
        """
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformanceRecord(
                strategy_name=strategy_name
            )

        self._performance[strategy_name].record_outcome(
            regime=regime,
            is_success=is_success,
            pnl=pnl,
        )

    def get_weights(
        self,
        regime: RegimeType,
        confidence: float = 1.0,
    ) -> dict[str, float]:
        """Calculate Bayesian-adapted weights for regime.

        Uses Beta distribution success probabilities to adjust base weights.

        Args:
            regime: Current market regime
            confidence: Regime detection confidence (0-1)

        Returns:
            Dict mapping strategy names to adapted weights
        """
        raw_weights: dict[str, float] = {}

        for name in self.strategy_names:
            record = self._performance.get(name)
            base = self.base_weights.get(name, 0.5)

            if record is None:
                raw_weights[name] = base
                continue

            # Get Bayesian success probability
            success_prob = record.get_success_probability(regime)

            # Get uncertainty (penalize uncertain strategies)
            uncertainty = record.get_uncertainty(regime)
            uncertainty_factor = 1.0 - (self.config.uncertainty_penalty * uncertainty)

            # Get recent performance bonus
            recent_win_rate = record.get_recent_win_rate()
            recent_bonus = (recent_win_rate - 0.5) * self.config.recent_bonus_weight

            # Combine factors
            bayesian_weight = base * success_prob * 2  # Scale to keep ~1.0 average
            adjusted = bayesian_weight * uncertainty_factor + recent_bonus

            raw_weights[name] = max(self.config.min_weight, adjusted)

        # Normalize weights
        total = sum(raw_weights.values())
        if total <= 0:
            equal = 1.0 / len(self.strategy_names) if self.strategy_names else 0.0
            return {name: equal for name in self.strategy_names}

        normalized = {name: w / total for name, w in raw_weights.items()}

        # Apply weight caps
        capped = {
            name: min(self.config.max_weight, max(self.config.min_weight, w))
            for name, w in normalized.items()
        }

        # Re-normalize after capping
        total = sum(capped.values())
        final = {name: w / total for name, w in capped.items()}

        # Apply smoothing with previous weights
        regime_key = regime.name
        if regime_key in self._current_weights:
            prev = self._current_weights[regime_key]
            smoothed = {
                name: (
                    self.config.smoothing * prev.get(name, final[name])
                    + (1 - self.config.smoothing) * final[name]
                )
                for name in final
            }
            # Re-normalize after smoothing
            total = sum(smoothed.values())
            final = {name: w / total for name, w in smoothed.items()}

        # Store current weights
        self._current_weights[regime_key] = final

        return final

    def get_strategy_performance(self, strategy_name: str) -> dict[str, Any]:
        """Get performance summary for a strategy.

        Args:
            strategy_name: Strategy to query

        Returns:
            Performance metrics dict
        """
        record = self._performance.get(strategy_name)
        if record is None:
            return {}

        return {
            "total_trades": record.total_trades,
            "total_wins": record.total_wins,
            "win_rate": record.total_wins / record.total_trades if record.total_trades > 0 else 0.0,
            "total_pnl": record.total_pnl,
            "recent_win_rate": record.get_recent_win_rate(),
            "regime_probabilities": {
                regime.name: record.get_success_probability(regime) for regime in RegimeType
            },
        }

    def get_all_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance summary for all strategies."""
        return {name: self.get_strategy_performance(name) for name in self.strategy_names}

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "strategy_names": self.strategy_names,
            "base_weights": self.base_weights,
            "config": {
                "prior_strength": self.config.prior_strength,
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "smoothing": self.config.smoothing,
                "uncertainty_penalty": self.config.uncertainty_penalty,
                "recent_bonus_weight": self.config.recent_bonus_weight,
            },
            "performance": {name: record.serialize() for name, record in self._performance.items()},
            "current_weights": self._current_weights,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> BayesianWeightUpdater:
        """Restore from serialized data."""
        config_data = data.get("config", {})
        config = BayesianWeightConfig(
            prior_strength=config_data.get("prior_strength", 2.0),
            min_weight=config_data.get("min_weight", 0.05),
            max_weight=config_data.get("max_weight", 0.8),
            smoothing=config_data.get("smoothing", 0.7),
            uncertainty_penalty=config_data.get("uncertainty_penalty", 0.3),
            recent_bonus_weight=config_data.get("recent_bonus_weight", 0.2),
        )

        updater = cls(
            strategy_names=data.get("strategy_names", []),
            base_weights=data.get("base_weights"),
            config=config,
        )

        # Restore performance records
        for name, record_data in data.get("performance", {}).items():
            updater._performance[name] = StrategyPerformanceRecord.deserialize(record_data)

        updater._current_weights = data.get("current_weights", {})

        return updater


__all__ = [
    "BayesianWeightConfig",
    "BayesianWeightUpdater",
    "StrategyPerformanceRecord",
]
