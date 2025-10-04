"""
Liquidity metrics tracker for rolling time-series analysis.

Manages per-symbol trade and spread data within configurable time windows,
providing volume and spread metrics for liquidity analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal


class LiquidityMetrics:
    """
    Rolling liquidity metrics calculator for a single symbol.

    Maintains rolling windows of volume, price impact, and depth
    for liquidity analysis.
    """

    def __init__(self, window_minutes: int = 15) -> None:
        """Initialize metrics with time window.

        Args:
            window_minutes: Rolling window duration in minutes
        """
        self.window_duration = timedelta(minutes=window_minutes)
        self._volume_data: list[tuple[datetime, Decimal]] = []  # (timestamp, volume)
        self._spread_data: list[tuple[datetime, Decimal]] = []  # (timestamp, spread_bps)
        self._trade_data: list[tuple[datetime, Decimal, Decimal]] = []  # (timestamp, price, size)

    def add_trade(self, price: Decimal, size: Decimal, timestamp: datetime | None = None) -> None:
        """Add trade data for volume calculation.

        Args:
            price: Trade price
            size: Trade size
            timestamp: Trade timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        notional = price * size
        self._volume_data.append((timestamp, notional))
        self._trade_data.append((timestamp, price, size))

        # Clean old data
        self._clean_old_data()

    def add_spread(self, spread_bps: Decimal, timestamp: datetime | None = None) -> None:
        """Add spread data.

        Args:
            spread_bps: Spread in basis points
            timestamp: Spread timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self._spread_data.append((timestamp, spread_bps))
        self._clean_old_data()

    def get_volume_metrics(self) -> dict[str, Decimal | int]:
        """Calculate volume metrics across multiple time windows.

        Returns:
            Dictionary with volume_1m, volume_5m, volume_15m, trade_count, avg_trade_size
        """
        if not self._volume_data:
            return {
                "volume_1m": Decimal("0"),
                "volume_5m": Decimal("0"),
                "volume_15m": Decimal("0"),
                "trade_count": 0,
                "avg_trade_size": Decimal("0"),
            }

        now = datetime.now()

        # Calculate volumes for different windows
        volumes: dict[str, Decimal] = {
            "1m": Decimal("0"),
            "5m": Decimal("0"),
            "15m": Decimal("0"),
        }
        windows = {"1m": 1, "5m": 5, "15m": 15}

        for window_name, minutes in windows.items():
            cutoff = now - timedelta(minutes=minutes)
            window_volume = sum(
                (volume for timestamp, volume in self._volume_data if timestamp >= cutoff),
                Decimal("0"),
            )
            volumes[window_name] = window_volume

        # Trade metrics
        trades_15m = [
            (ts, size) for ts, price, size in self._trade_data if ts >= now - timedelta(minutes=15)
        ]

        trade_count = len(trades_15m)
        avg_trade_size = Decimal("0")
        if trade_count > 0:
            total_size = sum((size for _, size in trades_15m), Decimal("0"))
            avg_trade_size = total_size / Decimal(trade_count)

        return {
            "volume_1m": volumes["1m"],
            "volume_5m": volumes["5m"],
            "volume_15m": volumes["15m"],
            "trade_count": trade_count,
            "avg_trade_size": avg_trade_size,
        }

    def get_spread_metrics(self) -> dict[str, Decimal]:
        """Calculate spread metrics over recent window.

        Returns spread statistics from last 5 minutes.

        Returns:
            Dictionary with avg_spread_bps, min_spread_bps, max_spread_bps
        """
        if not self._spread_data:
            return {
                "avg_spread_bps": Decimal("0"),
                "min_spread_bps": Decimal("0"),
                "max_spread_bps": Decimal("0"),
            }

        # Get spreads from last 5 minutes
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)

        recent_spreads = [spread for timestamp, spread in self._spread_data if timestamp >= cutoff]

        if not recent_spreads:
            return {
                "avg_spread_bps": Decimal("0"),
                "min_spread_bps": Decimal("0"),
                "max_spread_bps": Decimal("0"),
            }

        total_spread = sum(recent_spreads, Decimal("0"))
        count = Decimal(len(recent_spreads))
        return {
            "avg_spread_bps": total_spread / count if count > 0 else Decimal("0"),
            "min_spread_bps": min(recent_spreads),
            "max_spread_bps": max(recent_spreads),
        }

    def _clean_old_data(self) -> None:
        """Remove data older than window duration."""
        cutoff = datetime.now() - self.window_duration

        self._volume_data = [(ts, vol) for ts, vol in self._volume_data if ts >= cutoff]
        self._spread_data = [(ts, spread) for ts, spread in self._spread_data if ts >= cutoff]
        self._trade_data = [
            (ts, price, size) for ts, price, size in self._trade_data if ts >= cutoff
        ]


class MetricsTracker:
    """
    Tracks liquidity metrics across multiple symbols.

    Manages per-symbol LiquidityMetrics instances and provides
    a unified interface for trade/spread updates and metric retrieval.
    """

    def __init__(self, window_minutes: int = 15) -> None:
        """Initialize metrics tracker.

        Args:
            window_minutes: Rolling window duration for all symbols
        """
        self._window_minutes = window_minutes
        self._symbol_metrics: dict[str, LiquidityMetrics] = {}

    def _get_metrics(self, symbol: str) -> LiquidityMetrics:
        """Get or create metrics tracker for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            LiquidityMetrics instance for symbol
        """
        if symbol not in self._symbol_metrics:
            self._symbol_metrics[symbol] = LiquidityMetrics(window_minutes=self._window_minutes)
        return self._symbol_metrics[symbol]

    def add_trade(
        self, symbol: str, price: Decimal, size: Decimal, timestamp: datetime | None = None
    ) -> None:
        """Add trade data for symbol.

        Args:
            symbol: Trading symbol
            price: Trade price
            size: Trade size
            timestamp: Trade timestamp (defaults to now)
        """
        metrics = self._get_metrics(symbol)
        metrics.add_trade(price, size, timestamp)

    def add_spread(
        self, symbol: str, spread_bps: Decimal, timestamp: datetime | None = None
    ) -> None:
        """Add spread data for symbol.

        Args:
            symbol: Trading symbol
            spread_bps: Spread in basis points
            timestamp: Spread timestamp (defaults to now)
        """
        metrics = self._get_metrics(symbol)
        metrics.add_spread(spread_bps, timestamp)

    def get_volume_metrics(self, symbol: str) -> dict[str, Decimal | int]:
        """Get volume metrics for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with volume_1m, volume_5m, volume_15m, trade_count, avg_trade_size
        """
        metrics = self._get_metrics(symbol)
        return metrics.get_volume_metrics()

    def get_spread_metrics(self, symbol: str) -> dict[str, Decimal]:
        """Get spread metrics for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with avg_spread_bps, min_spread_bps, max_spread_bps
        """
        metrics = self._get_metrics(symbol)
        return metrics.get_spread_metrics()

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol has been tracked.

        Args:
            symbol: Trading symbol

        Returns:
            True if symbol has metrics, False otherwise
        """
        return symbol in self._symbol_metrics
