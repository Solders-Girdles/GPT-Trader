"""
Order book depth analyzer for liquidity assessment.

Analyzes order book structure including best bid/ask, spreads,
multi-level depth, and order flow imbalance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class DepthData:
    """Raw order book depth analysis data."""

    # Level 1 (best bid/ask)
    best_bid: Decimal
    best_ask: Decimal
    bid_size: Decimal
    ask_size: Decimal

    # Spread metrics
    spread: Decimal
    mid_price: Decimal
    spread_bps: Decimal

    # Multi-level depth (in base currency)
    bid_depth_1pct: Decimal
    ask_depth_1pct: Decimal
    bid_depth_5pct: Decimal
    ask_depth_5pct: Decimal
    bid_depth_10pct: Decimal
    ask_depth_10pct: Decimal

    # USD depth (depth * mid_price)
    depth_usd_1: Decimal
    depth_usd_5: Decimal
    depth_usd_10: Decimal

    # Imbalance metrics
    bid_ask_ratio: Decimal
    depth_imbalance: Decimal


class DepthAnalyzer:
    """
    Analyzes order book depth and structure.

    Extracts Level 1 data, calculates spreads, measures depth
    at multiple price levels, and computes imbalance metrics.
    """

    def analyze_depth(
        self,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        depth_thresholds: list[Decimal] | None = None,
    ) -> DepthData | None:
        """Analyze order book depth.

        Args:
            bids: Bid levels [(price, size), ...]
            asks: Ask levels [(price, size), ...]
            depth_thresholds: Price thresholds as fractions (default: [0.01, 0.05, 0.10])

        Returns:
            DepthData with analysis, or None if book is empty
        """
        if not bids or not asks:
            return None

        if depth_thresholds is None:
            depth_thresholds = [Decimal("0.01"), Decimal("0.05"), Decimal("0.10")]

        # Extract Level 1
        best_bid, bid_size = bids[0]
        best_ask, ask_size = asks[0]

        # Calculate spread
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else Decimal("10000")

        # Calculate depth at each threshold (using default [1%, 5%, 10%])
        depth_1pct = mid_price * depth_thresholds[0]
        depth_5pct = mid_price * depth_thresholds[1]
        depth_10pct = mid_price * depth_thresholds[2]

        bid_depth_1 = self._calculate_depth_in_range(bids, best_bid - depth_1pct, best_bid)
        ask_depth_1 = self._calculate_depth_in_range(asks, best_ask, best_ask + depth_1pct)

        bid_depth_5 = self._calculate_depth_in_range(bids, best_bid - depth_5pct, best_bid)
        ask_depth_5 = self._calculate_depth_in_range(asks, best_ask, best_ask + depth_5pct)

        bid_depth_10 = self._calculate_depth_in_range(bids, best_bid - depth_10pct, best_bid)
        ask_depth_10 = self._calculate_depth_in_range(asks, best_ask, best_ask + depth_10pct)

        # USD depth (notional value)
        depth_usd_1 = (bid_depth_1 + ask_depth_1) * mid_price
        depth_usd_5 = (bid_depth_5 + ask_depth_5) * mid_price
        depth_usd_10 = (bid_depth_10 + ask_depth_10) * mid_price

        # Imbalance metrics
        bid_ask_ratio = bid_size / ask_size if ask_size > 0 else Decimal("999")

        total_depth_5 = bid_depth_5 + ask_depth_5
        depth_imbalance = (
            ((bid_depth_5 - ask_depth_5) / total_depth_5) if total_depth_5 > 0 else Decimal("0")
        )

        return DepthData(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            mid_price=mid_price,
            spread_bps=spread_bps,
            bid_depth_1pct=bid_depth_1,
            ask_depth_1pct=ask_depth_1,
            bid_depth_5pct=bid_depth_5,
            ask_depth_5pct=ask_depth_5,
            bid_depth_10pct=bid_depth_10,
            ask_depth_10pct=ask_depth_10,
            depth_usd_1=depth_usd_1,
            depth_usd_5=depth_usd_5,
            depth_usd_10=depth_usd_10,
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=depth_imbalance,
        )

    def _calculate_depth_in_range(
        self,
        levels: list[tuple[Decimal, Decimal]],
        min_price: Decimal,
        max_price: Decimal,
    ) -> Decimal:
        """Calculate total size within price range.

        Args:
            levels: Order book levels [(price, size), ...]
            min_price: Minimum price (inclusive)
            max_price: Maximum price (inclusive)

        Returns:
            Total size within range
        """
        total = Decimal("0")
        for price, size in levels:
            if min_price <= price <= max_price:
                total += size
        return total
