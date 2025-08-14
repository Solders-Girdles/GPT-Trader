from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionMetrics:
    """Precise selection accuracy metrics."""

    top_k_accuracy: float
    rank_correlation: float
    regret: float
    turnover_rate: float
    slippage_cost: float


class SelectionAccuracyCalculator:
    """Calculate precise selection accuracy metrics."""

    def __init__(self, k: int = 3) -> None:
        self.k = int(k)

    def calculate_top_k_accuracy(
        self, predicted_ranks: list[str], actual_performance: dict[str, float]
    ) -> float:
        actual_ranks = sorted(actual_performance.items(), key=lambda x: x[1], reverse=True)
        actual_top_k = [strategy for strategy, _ in actual_ranks[: self.k]]
        predicted_top_k = predicted_ranks[: self.k]
        overlap = len(set(predicted_top_k) & set(actual_top_k))
        return overlap / max(self.k, 1)

    def calculate_rank_correlation(
        self, predicted_ranks: list[str], actual_performance: dict[str, float]
    ) -> float:
        predicted_rank_map = {strategy: i for i, strategy in enumerate(predicted_ranks)}
        actual_ranks = sorted(actual_performance.items(), key=lambda x: x[1], reverse=True)
        actual_rank_map = {strategy: i for i, (strategy, _) in enumerate(actual_ranks)}
        strategies = list(actual_performance.keys())
        pred_ranks = [predicted_rank_map.get(s, len(predicted_ranks)) for s in strategies]
        act_ranks = [actual_rank_map.get(s, len(actual_ranks)) for s in strategies]
        if len(pred_ranks) < 2 or len(act_ranks) < 2:
            return 0.0
        return float(np.corrcoef(pred_ranks, act_ranks)[0, 1])

    def calculate_regret(
        self,
        selected_strategies: list[str],
        actual_performance: dict[str, float],
        optimal_strategies: list[str],
    ) -> float:
        if not selected_strategies or not optimal_strategies:
            return 0.0
        selected_perf = np.mean([actual_performance.get(s, 0.0) for s in selected_strategies])
        optimal_perf = np.mean([actual_performance.get(s, 0.0) for s in optimal_strategies])
        return float(optimal_perf - selected_perf)
