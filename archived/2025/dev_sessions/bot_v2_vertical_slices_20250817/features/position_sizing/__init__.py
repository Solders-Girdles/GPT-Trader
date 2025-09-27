"""
Position Sizing Feature Slice

Intelligent position sizing with Kelly Criterion, confidence-based adjustments,
and market regime integration. Complete isolation maintained.
"""

from .position_sizing import (
    calculate_position_size,
    calculate_portfolio_allocation,
    PositionSizingResult
)

from .kelly import (
    kelly_criterion,
    fractional_kelly,
    KellyParameters
)

from .confidence import (
    confidence_adjusted_size,
    ConfidenceAdjustment
)

from .regime import (
    regime_adjusted_size,
    RegimeMultipliers
)

from .types import (
    PositionSizeRequest,
    PositionSizeResponse,
    RiskParameters,
    SizingMethod
)

__all__ = [
    # Main interface
    'calculate_position_size',
    'calculate_portfolio_allocation', 
    'PositionSizingResult',
    
    # Kelly Criterion
    'kelly_criterion',
    'fractional_kelly',
    'KellyParameters',
    
    # Confidence adjustments
    'confidence_adjusted_size',
    'ConfidenceAdjustment',
    
    # Regime adjustments
    'regime_adjusted_size',
    'RegimeMultipliers',
    
    # Types
    'PositionSizeRequest',
    'PositionSizeResponse', 
    'RiskParameters',
    'SizingMethod'
]