"""
Position Sizing Feature Slice

Intelligent position sizing with Kelly Criterion, confidence-based adjustments,
and market regime integration. Complete isolation maintained.
"""

from .confidence import ConfidenceAdjustment, confidence_adjusted_size
from .kelly import fractional_kelly, kelly_criterion
from .types import KellyParameters
from .position_sizing import (
    PositionSizingResult,
    calculate_portfolio_allocation,
    calculate_position_size,
)
from .regime import RegimeMultipliers, regime_adjusted_size
from .types import PositionSizeRequest, PositionSizeResponse, RiskParameters, SizingMethod

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
