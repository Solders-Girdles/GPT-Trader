"""
ML Strategy Selection feature slice - intelligent strategy selection.

Complete isolation - no external dependencies.
Week 1-2 of Path B: Smart Money implementation.
"""

from .ml_strategy import (
    train_strategy_selector,
    predict_best_strategy,
    evaluate_confidence,
    get_strategy_recommendation,
    backtest_with_ml,
    get_model_performance
)

from .types import (
    StrategyName,
    StrategyPrediction,
    MarketConditions,
    ModelPerformance,
    TrainingResult
)

__all__ = [
    # Core functions
    'train_strategy_selector',
    'predict_best_strategy',
    'evaluate_confidence',
    'get_strategy_recommendation',
    'backtest_with_ml',
    'get_model_performance',
    
    # Types
    'StrategyName',
    'StrategyPrediction',
    'MarketConditions',
    'ModelPerformance',
    'TrainingResult'
]