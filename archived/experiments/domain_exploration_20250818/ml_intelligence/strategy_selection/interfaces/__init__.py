"""
Strategy Selection Domain Interfaces.

This module exports the public API interfaces for the ML strategy selection domain.
All external consumers should import from this module to ensure stable interfaces
and proper encapsulation of implementation details.

Production Standards:
- Clean public API with comprehensive documentation
- Type safety with runtime validation
- Backwards compatibility guarantees
- Error handling with domain-specific exceptions
"""

from .types import (
    # Core types
    StrategyName,
    MarketRegime,
    ConfidenceLevel,
    
    # Data structures
    MarketConditions,
    StrategyPrediction,
    ModelPerformance,
    TrainingResult,
    StrategyPerformanceRecord,
    PredictionRequest,
    
    # Protocols
    StrategyPredictor,
    FeatureExtractor,
    
    # Exceptions
    StrategySelectionError,
    ModelNotTrainedError,
    InvalidMarketDataError,
    PredictionError,
    FeatureExtractionError,
)

# API functions are available at domain root level

__all__ = [
    # Types
    "StrategyName",
    "MarketRegime", 
    "ConfidenceLevel",
    "MarketConditions",
    "StrategyPrediction",
    "ModelPerformance",
    "TrainingResult",
    "StrategyPerformanceRecord",
    "PredictionRequest",
    
    # Protocols
    "StrategyPredictor",
    "FeatureExtractor",
    
    # Exceptions
    "StrategySelectionError",
    "ModelNotTrainedError",
    "InvalidMarketDataError",
    "PredictionError",
    "FeatureExtractionError",
]

# Version information
__version__ = "1.0.0"
__domain__ = "ml_intelligence.strategy_selection"
__last_updated__ = "2025-08-18"