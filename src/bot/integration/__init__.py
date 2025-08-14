"""Integration layer for GPT-Trader components.

This module provides bridge classes that connect different components
of the trading system, such as strategies and portfolio allocators.
"""

from .strategy_allocator_bridge import StrategyAllocatorBridge

__all__ = [
    "StrategyAllocatorBridge",
]
