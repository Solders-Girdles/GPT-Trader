"""
Market Regime Detection feature slice - intelligent market condition classification.

EXPERIMENTAL: Self-contained, uses synthetic/local data. Intended for
experimentation and demos; not used by the perps production path.
"""

from bot_v2.features.market_regime.market_regime import (
    analyze_regime_stability,
    detect_regime,
    get_regime_features,
    get_regime_history,
    monitor_regime_changes,
    predict_regime_change,
)
from bot_v2.features.market_regime.types import (
    MarketRegime,
    RegimeAnalysis,
    RegimeChangePrediction,
    RegimeFeatures,
    RegimeHistory,
    RegimeTransition,
    RiskSentiment,
    TrendRegime,
    VolatilityRegime,
)

__all__ = [
    # Core functions
    "detect_regime",
    "monitor_regime_changes",
    "get_regime_history",
    "predict_regime_change",
    "get_regime_features",
    "analyze_regime_stability",
    # Types
    "MarketRegime",
    "VolatilityRegime",
    "TrendRegime",
    "RiskSentiment",
    "RegimeAnalysis",
    "RegimeChangePrediction",
    "RegimeTransition",
    "RegimeHistory",
    "RegimeFeatures",
]

# Marker used by tooling and documentation
__experimental__ = True
