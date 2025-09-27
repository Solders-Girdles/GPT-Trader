"""
ML Intelligence Strategy Selection Domain.

This domain provides production-grade ML-driven strategy selection capabilities
for algorithmic trading systems. Features comprehensive model training,
validation, prediction, and monitoring with enterprise-level reliability.

Key Components:
- StrategySelector: Core ML model for strategy recommendations
- FeatureExtractor: Market data to ML feature transformation
- ConfidenceScorer: Prediction confidence assessment
- ValidationEngine: Model performance validation and testing

Key Features:
- Real-time strategy recommendations with confidence scoring
- Comprehensive validation including walk-forward analysis
- Feature engineering with quality assessment
- Model performance monitoring and degradation detection
- Production-ready error handling and logging

Usage Example:
    from domains.ml_intelligence.strategy_selection import (
        StrategySelector, FeatureExtractor, train_strategy_model
    )
    
    # Train model
    model = train_strategy_model(training_data)
    
    # Get predictions
    predictions = model.predict(market_conditions)
    best_strategy = predictions[0].strategy

Production Standards:
- >90% test coverage
- Complete type hints and validation
- Comprehensive error handling
- Structured logging
- Performance optimization
- Thread-safe operations
"""

from .core import StrategySelector, FeatureExtractor, ConfidenceScorer, ValidationEngine
from .interfaces.types import (
    StrategyName, MarketConditions, StrategyPrediction, ModelPerformance,
    TrainingResult, StrategyPerformanceRecord, PredictionRequest,
    MarketRegime, ConfidenceLevel, StrategyPredictor, FeatureExtractor as FeatureExtractorProtocol,
    # Error types
    StrategySelectionError, ModelNotTrainedError, InvalidMarketDataError,
    PredictionError, FeatureExtractionError
)
from .api import (
    train_strategy_model, get_strategy_recommendations, evaluate_model_performance,
    create_prediction_request, batch_predict_strategies
)

__version__ = "1.0.0"

__all__ = [
    # Core components
    "StrategySelector",
    "FeatureExtractor", 
    "ConfidenceScorer",
    "ValidationEngine",
    
    # Types
    "StrategyName",
    "MarketConditions",
    "StrategyPrediction",
    "ModelPerformance",
    "TrainingResult",
    "StrategyPerformanceRecord",
    "PredictionRequest",
    "MarketRegime",
    "ConfidenceLevel",
    "StrategyPredictor",
    "FeatureExtractorProtocol",
    
    # Error types
    "StrategySelectionError",
    "ModelNotTrainedError",
    "InvalidMarketDataError",
    "PredictionError",
    "FeatureExtractionError",
    
    # API functions
    "train_strategy_model",
    "get_strategy_recommendations",
    "evaluate_model_performance",
    "create_prediction_request",
    "batch_predict_strategies"
]