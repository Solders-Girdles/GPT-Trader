"""
Core ML Strategy Selection Module.

This module provides production-grade ML-driven strategy selection
capabilities for algorithmic trading systems.

Key Components:
- StrategySelector: Main ML model for strategy recommendations
- FeatureExtractor: Market data to ML feature transformation
- ConfidenceScorer: Prediction confidence assessment
- ValidationEngine: Model performance validation

Production Standards:
- Complete type hints and runtime validation
- Comprehensive error handling with specific exceptions
- Structured logging for all operations
- >90% test coverage requirement
- Cyclomatic complexity <10 per function
"""

from .strategy_selector import StrategySelector
from .feature_extractor import FeatureExtractor
from .confidence_scorer import ConfidenceScorer
from .validation_engine import ValidationEngine

__all__ = [
    "StrategySelector",
    "FeatureExtractor", 
    "ConfidenceScorer",
    "ValidationEngine"
]