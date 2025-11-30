"""
Market regime detection module.

Provides statistical classification of market conditions using O(1) algorithms:
- Trend detection via EMA analysis and ADX-style trend strength
- Volatility classification via ATR percentile and Bollinger squeeze
- Momentum scoring via rate of change
- Crisis detection for extreme conditions
- Probabilistic forecasting via regime transition matrix
"""

from gpt_trader.features.intelligence.regime.detector import MarketRegimeDetector
from gpt_trader.features.intelligence.regime.indicators import (
    OnlineATR,
    OnlineBollingerBands,
    OnlineTrendStrength,
    RegimeTransitionMatrix,
)
from gpt_trader.features.intelligence.regime.models import (
    RegimeConfig,
    RegimeState,
    RegimeType,
)

__all__ = [
    # Core detector
    "MarketRegimeDetector",
    # Models
    "RegimeConfig",
    "RegimeState",
    "RegimeType",
    # Advanced indicators (for direct use if needed)
    "OnlineATR",
    "OnlineBollingerBands",
    "OnlineTrendStrength",
    "RegimeTransitionMatrix",
]
