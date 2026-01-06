"""Hybrid strategy support for multi-venue trading.

This module provides base classes and types for strategies that can trade
across both spot and CFM futures markets.
"""

from .base import HybridStrategyBase
from .types import (
    HybridDecision,
    HybridMarketData,
    HybridStrategyConfig,
    TradingMode,
)

__all__ = [
    "HybridDecision",
    "HybridMarketData",
    "HybridStrategyBase",
    "HybridStrategyConfig",
    "TradingMode",
]
