"""Test helpers for bar runner tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from gpt_trader.backtesting.engine.bar_runner import IHistoricalDataProvider
from gpt_trader.core import Candle


def _create_mock_candle(
    symbol: str = "BTC-USD",
    ts: datetime | None = None,
    open_: Decimal = Decimal("50000"),
    high: Decimal = Decimal("50500"),
    low: Decimal = Decimal("49500"),
    close: Decimal = Decimal("50100"),
    volume: Decimal = Decimal("100"),
) -> Candle:
    _ = symbol  # Symbol is tracked externally in dict keys
    return Candle(
        ts=ts or datetime(2024, 1, 1, 12, 0, 0),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _create_mock_data_provider(
    candles_by_symbol: dict[str, list[Candle]] | None = None,
) -> MagicMock:
    provider = MagicMock(spec=IHistoricalDataProvider)

    async def mock_get_candles(
        symbol: str, granularity: str, start: datetime, end: datetime
    ) -> list[Candle]:
        _ = granularity
        if candles_by_symbol and symbol in candles_by_symbol:
            return [c for c in candles_by_symbol[symbol] if start <= c.ts < end]
        return []

    provider.get_candles = AsyncMock(side_effect=mock_get_candles)
    return provider


class _ChaosStub:
    def __init__(
        self,
        *,
        drop_candles: bool = False,
        latency: timedelta | None = None,
        ts_offset: timedelta | None = None,
    ) -> None:
        self._drop_candles = drop_candles
        self._latency = latency
        self._ts_offset = ts_offset

    def is_enabled(self) -> bool:
        return True

    def process_candle(
        self,
        symbol: str,
        candle: Candle,
        timestamp: datetime,
    ) -> Candle | None:
        _ = symbol, timestamp
        if self._drop_candles:
            return None
        if self._ts_offset:
            return Candle(
                ts=candle.ts + self._ts_offset,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
        return candle

    def apply_latency(self, timestamp: datetime) -> datetime:
        if self._latency:
            return timestamp + self._latency
        return timestamp
