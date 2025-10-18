"""Yahoo Finance-backed market data adapter for the new GPT-Trader seam."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from collections.abc import Callable

import pandas as pd

from bot_v2.data_providers import YFinanceProvider
from gpt_trader.domain import Bar

from .base import MarketData


def _as_decimal(value: object) -> Decimal:
    """Convert mixed numeric types (numpy, float, int) into ``Decimal`` safely."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return Decimal(str(value))


def _coerce_timestamp(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if hasattr(raw, "to_pydatetime"):
        return raw.to_pydatetime()  # type: ignore[call-arg]
    raise TypeError(f"Unsupported timestamp type: {type(raw)!r}")


def _period_from_lookback(lookback: int) -> str:
    """Map a simple lookback value to a yfinance period string."""
    return f"{max(lookback, 1)}d"


@dataclass
class YahooMarketData(MarketData):
    """Adapter that exposes the legacy ``YFinanceProvider`` via the new interface."""

    provider_factory: Callable[[], YFinanceProvider] = YFinanceProvider

    def __post_init__(self) -> None:
        self._provider = self.provider_factory()

    def bars(self, symbol: str, lookback: int, interval: str) -> Iterable[Bar]:
        period = _period_from_lookback(lookback)
        frame = self._provider.get_historical_data(symbol, period=period, interval=interval)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []

        # Trim to the requested number of bars while preserving order.
        trimmed = frame.tail(lookback)

        bars: list[Bar] = []
        for index, row in trimmed.iterrows():
            try:
                bar = Bar(
                    symbol=symbol.upper(),
                    timestamp=_coerce_timestamp(index),
                    open=_as_decimal(row["Open"]),
                    high=_as_decimal(row["High"]),
                    low=_as_decimal(row["Low"]),
                    close=_as_decimal(row["Close"]),
                    volume=_as_decimal(row["Volume"]),
                )
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise KeyError(f"Missing column in Yahoo data: {exc.args[0]!r}") from exc
            bars.append(bar)
        return bars


__all__ = ["YahooMarketData"]
