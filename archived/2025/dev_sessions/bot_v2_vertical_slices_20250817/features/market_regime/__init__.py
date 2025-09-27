"""
Market Regime Detection feature slice - intelligent market condition classification.

Complete isolation - no external dependencies.
Week 3 of Path B: Smart Money implementation.
"""

from .market_regime import (
    detect_regime,
    monitor_regime_changes,
    get_regime_history,
    predict_regime_change,
    get_regime_features,
    analyze_regime_stability
)

from .types import (
    MarketRegime,
    VolatilityRegime,
    TrendRegime,
    RiskSentiment,
    RegimeAnalysis,
    RegimeChangePrediction,
    RegimeTransition,
    RegimeHistory,
    RegimeFeatures
)

__all__ = [
    # Core functions
    'detect_regime',
    'monitor_regime_changes',
    'get_regime_history',
    'predict_regime_change',
    'get_regime_features',
    'analyze_regime_stability',
    
    # Types
    'MarketRegime',
    'VolatilityRegime', 
    'TrendRegime',
    'RiskSentiment',
    'RegimeAnalysis',
    'RegimeChangePrediction',
    'RegimeTransition',
    'RegimeHistory',
    'RegimeFeatures'
]