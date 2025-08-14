"""
Advanced Analytics Module for Phase 4.

This module provides comprehensive analytics capabilities for strategy analysis:
- Strategy Decomposition Analysis
- Performance Attribution
- Risk Decomposition
- Alpha Generation Analysis
"""

from .alpha_analysis import AlphaGenerationAnalyzer
from .attribution import PerformanceAttributionAnalyzer
from .decomposition import StrategyDecompositionAnalyzer
from .risk_decomposition import RiskDecompositionAnalyzer

__all__ = [
    "StrategyDecompositionAnalyzer",
    "PerformanceAttributionAnalyzer",
    "RiskDecompositionAnalyzer",
    "AlphaGenerationAnalyzer",
]
