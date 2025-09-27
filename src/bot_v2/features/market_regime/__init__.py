"""
Market Regime Detection feature slice - intelligent market condition classification.

EXPERIMENTAL: Self-contained, uses synthetic/local data. Intended for
experimentation and demos; not used by the perps production path.
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

# Marker used by tooling and documentation
__experimental__ = True
