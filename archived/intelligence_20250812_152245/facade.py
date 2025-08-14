from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .confidence_intervals import RobustConfidenceIntervals
from .metrics_registry import MetricsRegistry
from .order_simulator import L2SlippageModel
from .regime_labeling import RegimeLabeler
from .selection_metrics import SelectionAccuracyCalculator
from .transition_metrics import SlippageModel, TransitionSmoothnessCalculator


@dataclass
class Phase1IntelligenceToolkit:
    """Convenience wrapper for Phase 1 intelligence utilities.

    This façade keeps callsites compact while allowing direct access to
    underlying modules when more control is needed.
    """

    block_size: int = 20
    top_k: int = 3
    slippage_model: SlippageModel = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.ci = RobustConfidenceIntervals(block_size=self.block_size)
        self.selector_metrics = SelectionAccuracyCalculator(k=self.top_k)
        self.regimes = RegimeLabeler()
        self.slippage = self.slippage_model or L2SlippageModel()
        self.transition = TransitionSmoothnessCalculator(self.slippage)

    # Regime labeling
    def rule_based_regimes(self, df: pd.DataFrame) -> pd.Series:
        return self.regimes.create_rule_based_labels(df)

    def hmm_regimes(self, df: pd.DataFrame) -> pd.Series:
        return self.regimes.create_hmm_labels(df)

    # Selection metrics
    def selection_top_k_accuracy(
        self, predicted_ranks: list[str], actual_perf: dict[str, float]
    ) -> float:
        return self.selector_metrics.calculate_top_k_accuracy(predicted_ranks, actual_perf)

    def selection_rank_corr(
        self, predicted_ranks: list[str], actual_perf: dict[str, float]
    ) -> float:
        return self.selector_metrics.calculate_rank_correlation(predicted_ranks, actual_perf)

    def selection_regret(
        self, selected: list[str], actual_perf: dict[str, float], optimal: list[str]
    ) -> float:
        return self.selector_metrics.calculate_regret(selected, actual_perf, optimal)

    # Transition metrics
    def transition_smoothness(
        self,
        current_allocations: dict[str, float],
        target_allocations: dict[str, float],
        portfolio_value: float,
    ) -> float:
        return self.transition.calculate_smoothness_score(
            current_allocations, target_allocations, portfolio_value
        )

    # Confidence intervals
    def ci_compare(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        return self.ci.compare_ci_methods(returns, confidence_level)

    # Metrics registry
    @staticmethod
    def metrics_registry(path: Path) -> MetricsRegistry:
        return MetricsRegistry(path)


@dataclass
class ResearchDatasetBuilder:
    """Scaffolding to build leakage-safe datasets using a feature registry and pipeline.

    Example:
      builder = ResearchDatasetBuilder()
      X, y = builder.build(symbol_df, target_series, features=["basic", "regime"])
    """

    n_splits: int = 5

    def __post_init__(self) -> None:
        from .data_pipeline import (
            FeatureRegistry,
            LeakageFreePipeline,
            build_basic_features,
            build_calendar_features,
            build_microstructure_features,
            build_momentum_features,
            build_regime_features,
            build_return_features,
            build_trend_features,
            build_volatility_features,
            build_volume_features,
        )

        self.registry = FeatureRegistry()
        self.registry.register("basic", build_basic_features)
        self.registry.register("regime", build_regime_features)
        self.registry.register("returns", build_return_features)
        self.registry.register("volatility", build_volatility_features)
        self.registry.register("trend", build_trend_features)
        self.registry.register("momentum", build_momentum_features)
        self.registry.register("volume", build_volume_features)
        self.registry.register("microstructure", build_microstructure_features)
        self.registry.register("calendar", build_calendar_features)
        self.pipeline = LeakageFreePipeline(n_splits=self.n_splits)

    def build(
        self, df: pd.DataFrame, target: pd.Series, features: list[str]
    ) -> tuple[object, object]:
        base = pd.DataFrame({"Close": df["Close"]}) if "Close" in df.columns else df
        feat_df = self.registry.compose(features, base)
        X, y = self.pipeline.fit_transform(feat_df, target)
        return X, y
