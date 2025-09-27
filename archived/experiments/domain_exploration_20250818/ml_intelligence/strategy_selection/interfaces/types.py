"""
Type definitions for ML Strategy Selection domain.

This module provides comprehensive type definitions for strategy selection,
including market conditions, predictions, model performance metrics, and
configuration objects with strict validation.

Production Standards:
- Complete type hints with runtime validation
- Immutable data structures where appropriate
- Comprehensive error handling
- Full documentation coverage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Protocol, runtime_checkable
import numpy as np
from decimal import Decimal

# Configure module logger
logger = logging.getLogger(__name__)


class StrategyName(Enum):
    """
    Available trading strategies for ML selection.
    
    Each strategy represents a distinct trading approach with unique
    characteristics and market condition preferences.
    """
    SIMPLE_MA = "SimpleMAStrategy"
    MOMENTUM = "MomentumStrategy" 
    MEAN_REVERSION = "MeanReversionStrategy"
    VOLATILITY = "VolatilityStrategy"
    BREAKOUT = "BreakoutStrategy"
    RSI_DIVERGENCE = "RSIDivergenceStrategy"
    BOLLINGER_SQUEEZE = "BollingerSqueezeStrategy"
    
    def __str__(self) -> str:
        return self.value


class MarketRegime(Enum):
    """Market regime classifications for strategy selection."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_RANGE = "sideways_range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONAL = "transitional"
    CRISIS = "crisis"


class ConfidenceLevel(Enum):
    """Confidence level classifications for predictions."""
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"           # 75-90%
    MEDIUM = "medium"       # 50-75%
    LOW = "low"            # 25-50%
    VERY_LOW = "very_low"  # <25%


@dataclass(frozen=True)
class MarketConditions:
    """
    Comprehensive market conditions for strategy selection.
    
    Contains normalized market indicators and regime information
    required for ML model inference.
    
    Args:
        volatility: Annualized volatility percentage (0-100)
        trend_strength: Trend strength (-100 to 100, negative = downtrend)
        volume_ratio: Current vs average volume ratio (positive float)
        price_momentum: Rate of price change percentage
        market_regime: Current market regime classification
        vix_level: VIX fear index level
        correlation_spy: Correlation with SPY (-1 to 1)
        rsi: Relative Strength Index (0-100)
        bollinger_position: Position within Bollinger Bands (-1 to 1)
        atr_normalized: Average True Range normalized by price
        timestamp: When conditions were calculated
        
    Raises:
        ValueError: If any parameter is outside valid range
    """
    volatility: float
    trend_strength: float
    volume_ratio: float
    price_momentum: float
    market_regime: MarketRegime
    vix_level: float
    correlation_spy: float
    rsi: float = 50.0
    bollinger_position: float = 0.0
    atr_normalized: float = 0.02
    timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate market conditions parameters."""
        validations = [
            (0 <= self.volatility <= 200, f"Volatility must be 0-200, got {self.volatility}"),
            (-100 <= self.trend_strength <= 100, f"Trend strength must be -100 to 100, got {self.trend_strength}"),
            (self.volume_ratio > 0, f"Volume ratio must be positive, got {self.volume_ratio}"),
            (-100 <= self.price_momentum <= 100, f"Price momentum must be -100 to 100, got {self.price_momentum}"),
            (0 <= self.vix_level <= 100, f"VIX level must be 0-100, got {self.vix_level}"),
            (-1 <= self.correlation_spy <= 1, f"SPY correlation must be -1 to 1, got {self.correlation_spy}"),
            (0 <= self.rsi <= 100, f"RSI must be 0-100, got {self.rsi}"),
            (-2 <= self.bollinger_position <= 2, f"Bollinger position must be -2 to 2, got {self.bollinger_position}"),
            (self.atr_normalized >= 0, f"ATR normalized must be non-negative, got {self.atr_normalized}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Market conditions validation failed: {message}")
                raise ValueError(message)
        
        # Set timestamp if not provided
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.now())


@dataclass(frozen=True)
class StrategyPrediction:
    """
    ML model prediction for strategy performance.
    
    Contains expected performance metrics and confidence scores
    for a specific strategy under given market conditions.
    
    Args:
        strategy: Strategy being predicted
        expected_return: Predicted return percentage (annualized)
        confidence: Confidence score (0-1)
        predicted_sharpe: Predicted Sharpe ratio
        predicted_max_drawdown: Expected maximum drawdown (negative)
        ranking: Rank among all strategies (1-based)
        feature_importance: Feature importance scores
        model_version: Version of model used for prediction
        prediction_timestamp: When prediction was made
        
    Raises:
        ValueError: If any parameter is outside valid range
    """
    strategy: StrategyName
    expected_return: float
    confidence: float
    predicted_sharpe: float
    predicted_max_drawdown: float
    ranking: int
    feature_importance: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    prediction_timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate prediction parameters."""
        validations = [
            (-100 <= self.expected_return <= 200, f"Expected return must be -100 to 200%, got {self.expected_return}"),
            (0 <= self.confidence <= 1, f"Confidence must be 0-1, got {self.confidence}"),
            (-5 <= self.predicted_sharpe <= 10, f"Predicted Sharpe must be -5 to 10, got {self.predicted_sharpe}"),
            (-100 <= self.predicted_max_drawdown <= 0, f"Max drawdown must be negative, got {self.predicted_max_drawdown}"),
            (self.ranking >= 1, f"Ranking must be positive, got {self.ranking}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Strategy prediction validation failed: {message}")
                raise ValueError(message)
        
        # Set timestamp if not provided
        if self.prediction_timestamp is None:
            object.__setattr__(self, 'prediction_timestamp', datetime.now())
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    @property
    def risk_adjusted_score(self) -> float:
        """Calculate risk-adjusted performance score."""
        if self.predicted_sharpe <= 0:
            return 0.0
        return self.expected_return * self.confidence * min(self.predicted_sharpe / 2, 1.0)


@dataclass(frozen=True)
class ModelPerformance:
    """
    Comprehensive performance metrics for ML models.
    
    Tracks both statistical and financial performance metrics
    for model validation and monitoring.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mean_absolute_error: float
    r_squared: float
    backtest_correlation: float
    total_predictions: int
    successful_predictions: int
    out_of_sample_accuracy: Optional[float] = None
    calibration_score: Optional[float] = None
    feature_stability: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate performance metrics."""
        validations = [
            (0 <= self.accuracy <= 1, f"Accuracy must be 0-1, got {self.accuracy}"),
            (0 <= self.precision <= 1, f"Precision must be 0-1, got {self.precision}"),
            (0 <= self.recall <= 1, f"Recall must be 0-1, got {self.recall}"),
            (0 <= self.f1_score <= 1, f"F1 score must be 0-1, got {self.f1_score}"),
            (self.mean_absolute_error >= 0, f"MAE must be non-negative, got {self.mean_absolute_error}"),
            (-1 <= self.r_squared <= 1, f"R-squared must be -1 to 1, got {self.r_squared}"),
            (-1 <= self.backtest_correlation <= 1, f"Correlation must be -1 to 1, got {self.backtest_correlation}"),
            (self.total_predictions >= 0, f"Total predictions must be non-negative, got {self.total_predictions}"),
            (0 <= self.successful_predictions <= self.total_predictions, 
             f"Successful predictions must be <= total, got {self.successful_predictions}/{self.total_predictions}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Model performance validation failed: {message}")
                raise ValueError(message)
    
    @property
    def success_rate(self) -> float:
        """Calculate prediction success rate."""
        return self.successful_predictions / self.total_predictions if self.total_predictions > 0 else 0.0


@dataclass
class TrainingResult:
    """
    Results from ML model training process.
    
    Contains metadata about training process, hyperparameters,
    and performance metrics for model versioning and tracking.
    """
    model_id: str
    training_date: datetime
    features_used: List[str]
    training_samples: int
    validation_samples: int
    test_samples: int
    validation_score: float
    test_score: float
    best_hyperparameters: Dict[str, Any]
    training_time_seconds: float
    model_size_mb: float
    cross_validation_scores: Optional[List[float]] = None
    feature_selection_method: Optional[str] = None
    data_preprocessing_steps: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Validate training result parameters."""
        validations = [
            (len(self.model_id) > 0, "Model ID cannot be empty"),
            (len(self.features_used) > 0, "Features used cannot be empty"),
            (self.training_samples > 0, f"Training samples must be positive, got {self.training_samples}"),
            (self.validation_samples >= 0, f"Validation samples must be non-negative, got {self.validation_samples}"),
            (self.test_samples >= 0, f"Test samples must be non-negative, got {self.test_samples}"),
            (0 <= self.validation_score <= 1, f"Validation score must be 0-1, got {self.validation_score}"),
            (0 <= self.test_score <= 1, f"Test score must be 0-1, got {self.test_score}"),
            (self.training_time_seconds > 0, f"Training time must be positive, got {self.training_time_seconds}"),
            (self.model_size_mb > 0, f"Model size must be positive, got {self.model_size_mb}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Training result validation failed: {message}")
                raise ValueError(message)


@dataclass(frozen=True)
class StrategyPerformanceRecord:
    """
    Historical performance record for training data.
    
    Represents actual performance of a strategy under specific
    market conditions for supervised learning.
    """
    strategy: StrategyName
    symbol: str
    date: datetime
    market_conditions: MarketConditions
    actual_return: float
    actual_sharpe: float
    actual_drawdown: float
    trades_count: int
    win_rate: float
    holding_period_days: int = 1
    transaction_costs: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate performance record parameters."""
        validations = [
            (len(self.symbol) > 0, "Symbol cannot be empty"),
            (-100 <= self.actual_return <= 200, f"Return must be -100 to 200%, got {self.actual_return}"),
            (-10 <= self.actual_sharpe <= 10, f"Sharpe must be -10 to 10, got {self.actual_sharpe}"),
            (-100 <= self.actual_drawdown <= 0, f"Drawdown must be negative, got {self.actual_drawdown}"),
            (self.trades_count >= 0, f"Trades count must be non-negative, got {self.trades_count}"),
            (0 <= self.win_rate <= 1, f"Win rate must be 0-1, got {self.win_rate}"),
            (self.holding_period_days > 0, f"Holding period must be positive, got {self.holding_period_days}"),
            (self.transaction_costs >= 0, f"Transaction costs must be non-negative, got {self.transaction_costs}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Performance record validation failed: {message}")
                raise ValueError(message)


@dataclass
class PredictionRequest:
    """
    Request for strategy prediction.
    
    Contains all context needed for ML model to make
    strategy recommendations.
    """
    symbol: str
    market_conditions: MarketConditions
    lookback_days: int = 30
    top_n_strategies: int = 3
    min_confidence: float = 0.5
    risk_tolerance: float = 1.0
    exclude_strategies: Optional[List[StrategyName]] = None
    model_version: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate prediction request parameters."""
        validations = [
            (len(self.symbol) > 0, "Symbol cannot be empty"),
            (1 <= self.lookback_days <= 252, f"Lookback days must be 1-252, got {self.lookback_days}"),
            (1 <= self.top_n_strategies <= 10, f"Top N must be 1-10, got {self.top_n_strategies}"),
            (0 <= self.min_confidence <= 1, f"Min confidence must be 0-1, got {self.min_confidence}"),
            (0 <= self.risk_tolerance <= 3, f"Risk tolerance must be 0-3, got {self.risk_tolerance}")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Prediction request validation failed: {message}")
                raise ValueError(message)
        
        if self.exclude_strategies is None:
            self.exclude_strategies = []


@runtime_checkable
class StrategyPredictor(Protocol):
    """
    Protocol for strategy prediction models.
    
    Defines the interface that all strategy selection models
    must implement for consistent usage across the domain.
    """
    
    def predict(self, market_conditions: MarketConditions) -> List[StrategyPrediction]:
        """
        Predict best strategies for given market conditions.
        
        Args:
            market_conditions: Current market state
            
        Returns:
            List of strategy predictions sorted by performance
            
        Raises:
            ValueError: If market conditions are invalid
            RuntimeError: If model is not properly trained
        """
        ...
    
    def predict_confidence(self, strategy: StrategyName, market_conditions: MarketConditions) -> float:
        """
        Predict confidence for a specific strategy.
        
        Args:
            strategy: Strategy to evaluate
            market_conditions: Current market state
            
        Returns:
            Confidence score (0-1)
        """
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and performance information.
        
        Returns:
            Dictionary containing model version, performance metrics, etc.
        """
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """
    Protocol for feature extraction components.
    
    Defines interface for converting market conditions and
    raw data into ML-ready feature vectors.
    """
    
    def extract_features(self, market_conditions: MarketConditions) -> np.ndarray:
        """
        Extract numerical features from market conditions.
        
        Args:
            market_conditions: Market state to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        ...
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names in order
        """
        ...


# Error types for the domain
class StrategySelectionError(Exception):
    """Base exception for strategy selection domain."""
    pass


class ModelNotTrainedError(StrategySelectionError):
    """Raised when attempting to use an untrained model."""
    pass


class InvalidMarketDataError(StrategySelectionError):
    """Raised when market data is invalid or insufficient."""
    pass


class PredictionError(StrategySelectionError):
    """Raised when prediction fails."""
    pass


class FeatureExtractionError(StrategySelectionError):
    """Raised when feature extraction fails."""
    pass