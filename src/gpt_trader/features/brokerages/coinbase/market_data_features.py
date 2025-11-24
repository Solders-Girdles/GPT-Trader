"""Market data feature helpers for Coinbase WebSocket streams."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

__all__ = [
    "RollingWindow",
    "DepthSnapshot",
    "TradeTapeAgg",
    "get_expected_perps",
]

logger = get_logger(__name__, component="coinbase_market_data")


class RollingWindow:
    """Rolling time window for aggregating metrics like volume."""

    def __init__(self, duration_seconds: int) -> None:
        self.duration = duration_seconds
        self.values: deque[tuple[float, datetime]] = deque()
        self.sum = 0.0
        self.count = 0

    def add(self, value: float, timestamp: datetime | None = None) -> None:
        """Add value to rolling window."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.values.append((value, timestamp))
        self.sum += value
        self.count += 1

        # Remove old values outside window
        self._cleanup(timestamp)

    def _cleanup(self, current_time: datetime) -> None:
        """Remove values outside the time window."""
        cutoff = current_time - timedelta(seconds=self.duration)

        while self.values and self.values[0][1] < cutoff:
            old_value, _ = self.values.popleft()
            self.sum -= old_value
            self.count -= 1

    def get_stats(self) -> dict[str, float]:
        """Get current window statistics."""
        return {
            "sum": self.sum,
            "count": self.count,
            "avg": self.sum / self.count if self.count > 0 else 0.0,
        }


class DepthSnapshot:
    """Order book depth snapshot with spread/depth calculations."""

    def __init__(self, levels: list[tuple[Decimal, Decimal, str]]) -> None:
        """
        Initialize with order book levels.

        Args:
            levels: List of (price, size, side) tuples
        """
        self.bids = []
        self.asks = []

        # Sort levels into bids and asks
        for price, size, side in levels:
            if side.lower() in ("buy", "bid"):
                self.bids.append((price, size))
            elif side.lower() in ("sell", "ask"):
                self.asks.append((price, size))

        # Sort bids descending, asks ascending
        self.bids.sort(key=lambda x: x[0], reverse=True)
        self.asks.sort(key=lambda x: x[0])

    @property
    def mid(self) -> Decimal | None:
        """Get mid price."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        if not self.bids or not self.asks:
            return 0.0

        bid = self.bids[0][0]
        ask = self.asks[0][0]

        if bid <= 0:
            return 0.0

        return float((ask - bid) / bid * 10000)

    def get_depth(self, levels: int = 1) -> tuple[Decimal, Decimal]:
        """
        Get depth at specified number of levels.

        Returns:
            Tuple of (bid_depth, ask_depth)
        """
        bid_depth = sum((size for _, size in self.bids[:levels]), Decimal("0"))
        ask_depth = sum((size for _, size in self.asks[:levels]), Decimal("0"))
        return bid_depth, ask_depth

    def get_l1_depth(self) -> Decimal:
        """Get L1 (top-of-book) depth."""
        bid_depth, ask_depth = self.get_depth(1)
        return min(bid_depth, ask_depth)

    def get_l10_depth(self) -> Decimal:
        """Get L10 (top 10 levels) depth."""
        bid_depth, ask_depth = self.get_depth(10)
        return bid_depth + ask_depth


class TradeTapeAgg:
    """Trade tape aggregator for volume/size/aggressor analysis."""

    def __init__(self, duration_seconds: int) -> None:
        self.duration = duration_seconds
        self.trades: deque[dict[str, Any]] = deque()

    def add_trade(
        self, price: Decimal, size: Decimal, side: str, timestamp: datetime | None = None
    ) -> None:
        """Add trade to aggregator."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        trade = {
            "price": price,
            "size": size,
            "side": side.lower(),
            "timestamp": timestamp,
            "notional": price * size,
        }

        self.trades.append(trade)
        self._cleanup(timestamp)

    def _cleanup(self, current_time: datetime) -> None:
        """Remove trades outside time window."""
        cutoff = current_time - timedelta(seconds=self.duration)

        while self.trades and self.trades[0]["timestamp"] < cutoff:
            self.trades.popleft()

    def get_vwap(self) -> Decimal:
        """Get volume-weighted average price."""
        if not self.trades:
            return Decimal("0")

        total_notional = sum((trade["notional"] for trade in self.trades), Decimal("0"))
        total_volume = sum((trade["size"] for trade in self.trades), Decimal("0"))

        if total_volume == 0:
            return Decimal("0")

        return total_notional / total_volume

    def get_avg_size(self) -> Decimal:
        """Get average trade size."""
        if not self.trades:
            return Decimal("0")

        total_volume = sum((trade["size"] for trade in self.trades), Decimal("0"))
        return total_volume / Decimal(len(self.trades))

    def get_aggressor_ratio(self) -> float:
        """Get ratio of buy-side (aggressive) trades."""
        if not self.trades:
            return 0.0

        buy_trades = sum(1 for trade in self.trades if trade["side"] == "buy")
        return buy_trades / len(self.trades)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive trade statistics."""
        return {
            "count": len(self.trades),
            "volume": sum((trade["size"] for trade in self.trades), Decimal("0")),
            "vwap": self.get_vwap(),
            "avg_size": self.get_avg_size(),
            "aggressor_ratio": self.get_aggressor_ratio(),
        }


def get_expected_perps() -> set:
    """Get expected perpetuals symbols for validation."""
    return {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}
