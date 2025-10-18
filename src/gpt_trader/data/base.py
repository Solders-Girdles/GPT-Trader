"""Abstract interfaces for data providers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from gpt_trader.domain import Bar


class MarketData(Protocol):
    """Interface for retrieving market data series."""

    def bars(self, symbol: str, lookback: int, interval: str) -> Iterable[Bar]:
        """Yield OHLCV bars for ``symbol``."""
        raise NotImplementedError


__all__ = ["MarketData"]
