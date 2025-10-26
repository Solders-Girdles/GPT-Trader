"""Compatibility shim for legacy imports of LiquidityService."""

from __future__ import annotations

from .liquidity import (
    DepthAnalysis,
    ImpactEstimate,
    LiquidityCondition,
    LiquidityMetrics,
    LiquidityService,
    OrderBookLevel,
    create_liquidity_service,
)

__all__ = [
    "LiquidityService",
    "create_liquidity_service",
    "LiquidityMetrics",
    "DepthAnalysis",
    "ImpactEstimate",
    "LiquidityCondition",
    "OrderBookLevel",
]
