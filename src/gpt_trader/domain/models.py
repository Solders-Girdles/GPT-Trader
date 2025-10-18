"""Core domain models used across GPT-Trader modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class Bar:
    """Single OHLCV candle."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy output prior to broker execution."""

    symbol: str
    action: Action
    confidence: float = 1.0
    size: float | None = None
    metadata: dict[str, Any] | None = None


class Strategy(Protocol):
    """Minimal strategy interface for deterministic components."""

    def decide(self, bars: list[Bar]) -> Signal: ...


__all__ = ["Bar", "Signal", "Strategy", "Action"]
