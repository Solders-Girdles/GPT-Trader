"""
Type definitions for ML strategy selection - LOCAL to this slice.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class StrategyName(Enum):
    """Available strategies for ML selection."""
    SIMPLE_MA = "SimpleMAStrategy"
    MOMENTUM = "MomentumStrategy"
    MEAN_REVERSION = "MeanReversionStrategy"
    VOLATILITY = "VolatilityStrategy"
    BREAKOUT = "BreakoutStrategy"


@dataclass
class MarketConditions:
    """Current market conditions for strategy selection."""
    volatility: float  # 0-100 scale
    trend_strength: float  # -100 to 100 (negative = downtrend)
    volume_ratio: float  # Current vs average volume
    price_momentum: float  # Rate of change
    market_regime: str  # 'bull', 'bear', 'sideways'
    vix_level: float  # Fear index
    correlation_spy: float  # Correlation with market
    


@dataclass
class StrategyPrediction:
    """ML model prediction for strategy performance."""
    strategy: StrategyName
    expected_return: float  # Predicted return %
    confidence: float  # 0-1 confidence score
    predicted_sharpe: float  # Risk-adjusted return
    predicted_max_drawdown: float  # Expected max loss
    ranking: int  # Rank among all strategies
    
    
@dataclass
class ModelPerformance:
    """Performance metrics for the ML model."""
    accuracy: float  # Classification accuracy
    precision: float  # Precision score
    recall: float  # Recall score
    f1_score: float  # F1 score
    mean_absolute_error: float  # For return prediction
    r_squared: float  # For regression quality
    backtest_correlation: float  # Predicted vs actual
    total_predictions: int
    successful_predictions: int
    

@dataclass
class TrainingResult:
    """Result from training the ML model."""
    model_id: str
    training_date: datetime
    features_used: List[str]
    training_samples: int
    validation_score: float
    test_score: float
    best_hyperparameters: Dict[str, any]
    training_time_seconds: float
    model_size_mb: float


@dataclass 
class StrategyPerformanceRecord:
    """Historical performance record for training."""
    strategy: StrategyName
    symbol: str
    date: datetime
    market_conditions: MarketConditions
    actual_return: float
    actual_sharpe: float
    actual_drawdown: float
    trades_count: int
    win_rate: float