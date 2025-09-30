"""
Position Sizing Feature Slice

Intelligent position sizing with Kelly Criterion, confidence-based adjustments,
and market regime integration. Complete isolation maintained.
"""

from bot_v2.features.position_sizing.confidence import (
    ConfidenceAdjustment,
    confidence_adjusted_size,
)
from bot_v2.features.position_sizing.kelly import fractional_kelly, kelly_criterion
from bot_v2.features.position_sizing.position_sizing import (
    PositionSizingResult,
    calculate_portfolio_allocation,
    calculate_position_size,
)
from bot_v2.features.position_sizing.regime import RegimeMultipliers, regime_adjusted_size
from bot_v2.features.position_sizing.types import (
    KellyParameters,
    PositionSizeRequest,
    PositionSizeResponse,
    RiskParameters,
    SizingMethod,
)

__all__ = [
    # Main interface
    "calculate_position_size",
    "calculate_portfolio_allocation",
    "PositionSizingResult",
    # Kelly Criterion
    "kelly_criterion",
    "fractional_kelly",
    "KellyParameters",
    # Confidence adjustments
    "confidence_adjusted_size",
    "ConfidenceAdjustment",
    # Regime adjustments
    "regime_adjusted_size",
    "RegimeMultipliers",
    # Types
    "PositionSizeRequest",
    "PositionSizeResponse",
    "RiskParameters",
    "SizingMethod",
]
