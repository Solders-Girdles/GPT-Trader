from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class LiquidityCondition(Enum):
    """Market liquidity conditions."""

    EXCELLENT = "excellent"  # Deep, tight markets
    GOOD = "good"  # Normal liquidity
    FAIR = "fair"  # Moderate liquidity concerns
    POOR = "poor"  # Shallow, wide markets
    CRITICAL = "critical"  # Very poor liquidity


@dataclass
class OrderBookLevel:
    """Single order book level."""

    price: Decimal
    size: Decimal
    cumulative_size: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.cumulative_size = self.size


@dataclass
class DepthAnalysis:
    """Order book depth analysis."""

    symbol: str
    timestamp: datetime

    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    spread: Decimal
    spread_bps: Decimal

    depth_usd_1: Decimal
    depth_usd_5: Decimal
    depth_usd_10: Decimal

    bid_ask_ratio: Decimal
    depth_imbalance: Decimal

    liquidity_score: Decimal
    condition: LiquidityCondition

    def to_dict(self) -> dict[str, object]:  # Serialization helper
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_price": float(self.bid_price),
            "ask_price": float(self.ask_price),
            "bid_size": float(self.bid_size),
            "ask_size": float(self.ask_size),
            "spread": float(self.spread),
            "spread_bps": float(self.spread_bps),
            "depth_usd_1": float(self.depth_usd_1),
            "depth_usd_5": float(self.depth_usd_5),
            "depth_usd_10": float(self.depth_usd_10),
            "bid_ask_ratio": float(self.bid_ask_ratio),
            "depth_imbalance": float(self.depth_imbalance),
            "liquidity_score": float(self.liquidity_score),
            "condition": self.condition.value,
        }


@dataclass
class ImpactEstimate:
    """Market impact estimate for order execution."""

    symbol: str
    side: str
    quantity: Decimal
    estimated_impact_bps: Decimal
    estimated_avg_price: Decimal
    max_impact_price: Decimal
    slippage_cost: Decimal
    recommended_slicing: bool
    max_slice_size: Decimal | None
    use_post_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "estimated_impact_bps": float(self.estimated_impact_bps),
            "estimated_avg_price": float(self.estimated_avg_price),
            "max_impact_price": float(self.max_impact_price),
            "slippage_cost": float(self.slippage_cost),
            "recommended_slicing": self.recommended_slicing,
            "max_slice_size": float(self.max_slice_size) if self.max_slice_size else None,
            "use_post_only": self.use_post_only,
        }


__all__ = [
    "LiquidityCondition",
    "OrderBookLevel",
    "DepthAnalysis",
    "ImpactEstimate",
]
