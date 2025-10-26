from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Tuple

from bot_v2.utilities.logging_patterns import get_logger

from .utils import ensure_utc_aware, utc_now

logger = get_logger(__name__, component="liquidity_metrics")


class LiquidityMetrics:
    """
    Rolling liquidity metrics calculator.

    Maintains rolling windows of volume, price impact, and depth for liquidity analysis.
    """

    def __init__(self, window_minutes: int = 15) -> None:
        self.window_duration = timedelta(minutes=window_minutes)
        self._volume_data: list[tuple[datetime, Decimal]] = []
        self._spread_data: list[tuple[datetime, Decimal]] = []
        self._trade_data: list[tuple[datetime, Decimal, Decimal]] = []

    def add_trade(self, price: Decimal, size: Decimal, timestamp: datetime | None = None) -> None:
        """Add trade data for volume calculation."""
        ts = utc_now() if timestamp is None else ensure_utc_aware(timestamp)

        notional = price * size
        self._volume_data.append((ts, notional))
        self._trade_data.append((ts, price, size))
        self._clean_old_data()

    def add_spread(self, spread_bps: Decimal, timestamp: datetime | None = None) -> None:
        """Add spread data."""
        ts = utc_now() if timestamp is None else ensure_utc_aware(timestamp)

        self._spread_data.append((ts, spread_bps))
        self._clean_old_data()

    def get_volume_metrics(self) -> dict[str, Decimal | int]:
        """Calculate volume metrics."""
        if not self._volume_data:
            return {
                "volume_1m": Decimal("0"),
                "volume_5m": Decimal("0"),
                "volume_15m": Decimal("0"),
                "trade_count": 0,
                "avg_trade_size": Decimal("0"),
            }

        now = utc_now()
        windows = {"1m": 1, "5m": 5, "15m": 15}
        volumes: dict[str, Decimal] = {}

        for window_name, minutes in windows.items():
            cutoff = now - timedelta(minutes=minutes)
            volume = sum(
                volume
                for timestamp, volume in self._volume_data
                if ensure_utc_aware(timestamp) >= cutoff
            )
            volumes[window_name] = volume

        trades_15m = [
            (ts, size)
            for ts, _price, size in self._trade_data
            if ensure_utc_aware(ts) >= now - timedelta(minutes=15)
        ]
        trade_count = len(trades_15m)
        avg_trade_size = (
            sum((size for _ts, size in trades_15m), Decimal("0")) / Decimal(trade_count)
            if trade_count
            else Decimal("0")
        )

        return {
            "volume_1m": volumes.get("1m", Decimal("0")),
            "volume_5m": volumes.get("5m", Decimal("0")),
            "volume_15m": volumes.get("15m", Decimal("0")),
            "trade_count": trade_count,
            "avg_trade_size": avg_trade_size,
        }

    def get_spread_metrics(self) -> dict[str, Decimal]:
        """Calculate spread metrics."""
        if not self._spread_data:
            return {"avg_spread_bps": Decimal("0"), "min_spread_bps": Decimal("0"), "max_spread_bps": Decimal("0")}

        now = utc_now()
        cutoff = now - timedelta(minutes=5)

        recent_spreads = [
            spread
            for timestamp, spread in self._spread_data
            if ensure_utc_aware(timestamp) >= cutoff
        ]

        if not recent_spreads:
            return {"avg_spread_bps": Decimal("0"), "min_spread_bps": Decimal("0"), "max_spread_bps": Decimal("0")}

        total_spread = sum(recent_spreads, Decimal("0"))
        count = Decimal(len(recent_spreads))

        return {
            "avg_spread_bps": total_spread / count if count > 0 else Decimal("0"),
            "min_spread_bps": min(recent_spreads),
            "max_spread_bps": max(recent_spreads),
        }

    def _clean_old_data(self) -> None:
        """Remove data older than window duration."""
        cutoff = utc_now() - self.window_duration

        def _filter(data: Iterable[Tuple], transformer):
            return [
                transformer(ts, *rest)
                for ts, *rest in data
                if ensure_utc_aware(ts) >= cutoff
            ]

        self._volume_data = _filter(self._volume_data, lambda ts, vol: (ensure_utc_aware(ts), vol))
        self._spread_data = _filter(self._spread_data, lambda ts, spread: (ensure_utc_aware(ts), spread))
        self._trade_data = _filter(
            self._trade_data, lambda ts, price, size: (ensure_utc_aware(ts), price, size)
        )


__all__ = ["LiquidityMetrics"]
