"""Hybrid strategy support for multi-venue trading.

This module provides base classes and types for strategies that can trade
across both spot and CFM futures markets.
"""

from .types import (
    HybridDecision,
    HybridMarketData,
    HybridStrategyConfig,
    TradingMode,
)
from .base import HybridStrategyBase

__all__ = [
    "HybridDecision",
    "HybridMarketData",
    "HybridStrategyBase",
    "HybridStrategyConfig",
    "TradingMode",
]
