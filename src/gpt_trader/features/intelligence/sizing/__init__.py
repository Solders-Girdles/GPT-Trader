"""
Regime-aware position sizing module.

Provides intelligent position sizing based on:
- Market regime (scale down in crisis, adjust for volatility)
- ATR-based volatility scaling
- Kelly criterion for optimal sizing
- Risk budgeting across strategies
"""

from gpt_trader.features.intelligence.sizing.position_sizer import (
    PositionSizer,
    PositionSizingConfig,
    SizingResult,
)

__all__ = [
    "PositionSizer",
    "PositionSizingConfig",
    "SizingResult",
]
