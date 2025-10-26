from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional

from bot_v2.features.brokerages.core.interfaces import Candle, Quote


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _derive_quote(symbol: str, candle: Candle) -> Quote:
    return Quote(
        symbol=symbol,
        bid=candle.close * Decimal("0.9995"),
        ask=candle.close * Decimal("1.0005"),
        last=candle.close,
        ts=candle.ts,
    )


@dataclass
class MarketState:
    current_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    bars: Dict[str, Candle] = field(default_factory=dict)
    quotes: Dict[str, Quote] = field(default_factory=dict)
    next_bars: Dict[str, Candle] = field(default_factory=dict)

    def update(
        self,
        current_time: datetime,
        bars: Dict[str, Candle],
        quotes: Optional[Dict[str, Quote]] = None,
        next_bars: Optional[Dict[str, Candle]] = None,
    ) -> None:
        self.current_time = _ensure_utc(current_time)
        self.bars = bars
        self.next_bars = next_bars or {}
        if quotes is not None:
            self.quotes = quotes
        else:
            self.quotes = {symbol: _derive_quote(symbol, candle) for symbol, candle in bars.items()}

    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self.quotes:
            raise KeyError(f"Quote not available for {symbol}")
        return self.quotes[symbol]

    def get_bar(self, symbol: str) -> Optional[Candle]:
        return self.bars.get(symbol)

    def get_next_bar(self, symbol: str) -> Optional[Candle]:
        return self.next_bars.get(symbol)


__all__ = ["MarketState"]
