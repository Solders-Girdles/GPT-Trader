"""
ML Strategy Selection feature slice - intelligent strategy selection.

EXPERIMENTAL: Self-contained, uses synthetic/local data and local models.
Useful for experimentation, not in the perps critical path.
"""

from bot_v2.features.ml_strategy.ml_strategy import (
    backtest_with_ml,
    evaluate_confidence,
    get_model_performance,
    get_strategy_recommendation,
    predict_best_strategy,
    train_strategy_selector,
)
from bot_v2.features.ml_strategy.types import (
    MarketConditions,
    ModelPerformance,
    StrategyName,
    StrategyPrediction,
    TrainingResult,
)

__all__ = [
    # Core functions
    "train_strategy_selector",
    "predict_best_strategy",
    "evaluate_confidence",
    "get_strategy_recommendation",
    "backtest_with_ml",
    "get_model_performance",
    # Types
    "StrategyName",
    "StrategyPrediction",
    "MarketConditions",
    "ModelPerformance",
    "TrainingResult",
]

# Marker used by tooling and documentation
__experimental__ = True
