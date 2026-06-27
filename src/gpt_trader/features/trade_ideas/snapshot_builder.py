"""Build proposer-safe market snapshots from historical candle sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from gpt_trader.core import Candle
from gpt_trader.features.trade_ideas.snapshot import (
    MarketSnapshot,
    SnapshotIntegrityError,
    SymbolSeries,
)


class HistoricalCandleSource(Protocol):
    """Read-only source for historical candles."""

    async def fetch_candles(
        self,
        *,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        """Return candles for ``symbol`` in ``[start, end)``."""


@dataclass(frozen=True, slots=True)
class MarketSnapshotBuildRequest:
    """Configuration for one point-in-time snapshot build."""

    symbols: tuple[str, ...]
    granularity: str
    lookback: int
    as_of: datetime

    def __post_init__(self) -> None:
        if not self.symbols:
            raise SnapshotIntegrityError(
                "Market snapshot build requires at least one symbol",
                field="symbols",
            )
        if any(not symbol.strip() for symbol in self.symbols):
            raise SnapshotIntegrityError(
                "Market snapshot symbols must be non-empty",
                field="symbols",
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise SnapshotIntegrityError(
                "Market snapshot symbols must be unique",
                field="symbols",
            )
        if self.lookback < 1:
            raise SnapshotIntegrityError(
                "Market snapshot lookback must be at least 1",
                field="lookback",
            )
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise SnapshotIntegrityError(
                "Market snapshot as_of must include a timezone",
                field="as_of",
            )
        if granularity_duration(self.granularity) is None:
            raise SnapshotIntegrityError(
                f"Unsupported snapshot granularity: {self.granularity}",
                field="granularity",
            )


class MarketSnapshotBuilder:
    """Build ``MarketSnapshot`` objects without account or execution access."""

    def __init__(
        self,
        candle_source: HistoricalCandleSource,
        *,
        source_label: str = "coinbase:market-candles",
    ) -> None:
        if not source_label.strip():
            raise SnapshotIntegrityError(
                "Market snapshot source label must be non-empty",
                field="source",
            )
        self._candle_source = candle_source
        self._source_label = source_label

    async def build(self, request: MarketSnapshotBuildRequest) -> MarketSnapshot:
        """Fetch completed historical candles and build a point-in-time snapshot."""
        duration = granularity_duration(request.granularity)
        if duration is None:
            raise SnapshotIntegrityError(
                f"Unsupported snapshot granularity: {request.granularity}",
                field="granularity",
            )
        as_of = _normalize_utc(request.as_of)
        start = as_of - duration * request.lookback

        series: list[SymbolSeries] = []
        for symbol in request.symbols:
            candles = await self._candle_source.fetch_candles(
                symbol=symbol,
                granularity=request.granularity,
                start=start,
                end=as_of,
            )
            completed = _completed_window(
                symbol=symbol,
                candles=candles,
                start=start,
                as_of=as_of,
                duration=duration,
                lookback=request.lookback,
            )
            series.append(
                SymbolSeries(
                    symbol=symbol,
                    granularity=request.granularity,
                    candles=completed,
                )
            )

        return MarketSnapshot(
            as_of=as_of,
            source=_source_metadata(
                self._source_label,
                granularity=request.granularity,
                lookback=request.lookback,
                as_of=as_of,
            ),
            series=tuple(series),
        )


def market_snapshot_to_payload(snapshot: MarketSnapshot) -> dict[str, Any]:
    """Serialize a ``MarketSnapshot`` into the fixture shape accepted by the CLI."""
    return {
        "as_of": snapshot.as_of.isoformat(),
        "source": snapshot.source,
        "series": [
            {
                "symbol": symbol_series.symbol,
                "granularity": symbol_series.granularity,
                "candles": [
                    {
                        "ts": candle.ts.isoformat(),
                        "open": str(candle.open),
                        "high": str(candle.high),
                        "low": str(candle.low),
                        "close": str(candle.close),
                        "volume": str(candle.volume),
                    }
                    for candle in symbol_series.candles
                ],
            }
            for symbol_series in snapshot.series
        ],
    }


def granularity_duration(granularity: str) -> timedelta | None:
    """Return the duration for supported Coinbase candle granularities."""
    normalized = granularity.strip().upper().replace("-", "_")
    return {
        "1M": timedelta(minutes=1),
        "1MIN": timedelta(minutes=1),
        "1MINUTE": timedelta(minutes=1),
        "ONE_MINUTE": timedelta(minutes=1),
        "5M": timedelta(minutes=5),
        "5MIN": timedelta(minutes=5),
        "5MINUTE": timedelta(minutes=5),
        "FIVE_MINUTE": timedelta(minutes=5),
        "15M": timedelta(minutes=15),
        "15MIN": timedelta(minutes=15),
        "15MINUTE": timedelta(minutes=15),
        "FIFTEEN_MINUTE": timedelta(minutes=15),
        "30M": timedelta(minutes=30),
        "30MIN": timedelta(minutes=30),
        "30MINUTE": timedelta(minutes=30),
        "THIRTY_MINUTE": timedelta(minutes=30),
        "1H": timedelta(hours=1),
        "1HR": timedelta(hours=1),
        "1HOUR": timedelta(hours=1),
        "ONE_HOUR": timedelta(hours=1),
        "2H": timedelta(hours=2),
        "2HR": timedelta(hours=2),
        "2HOUR": timedelta(hours=2),
        "TWO_HOUR": timedelta(hours=2),
        "6H": timedelta(hours=6),
        "6HR": timedelta(hours=6),
        "6HOUR": timedelta(hours=6),
        "SIX_HOUR": timedelta(hours=6),
        "1D": timedelta(days=1),
        "1DAY": timedelta(days=1),
        "ONE_DAY": timedelta(days=1),
    }.get(normalized)


def _completed_window(
    *,
    symbol: str,
    candles: Sequence[Candle],
    start: datetime,
    as_of: datetime,
    duration: timedelta,
    lookback: int,
) -> tuple[Candle, ...]:
    normalized = tuple(_normalize_candle(candle) for candle in candles)
    _require_ascending(symbol, normalized)
    completed = tuple(
        candle for candle in normalized if start <= candle.ts and candle.ts + duration <= as_of
    )
    if not completed:
        raise SnapshotIntegrityError(
            f"No completed candles for '{symbol}' before snapshot as_of {as_of.isoformat()}",
            field="candles",
        )
    return completed[-lookback:]


def _normalize_candle(candle: Candle) -> Candle:
    return Candle(
        ts=_normalize_utc(candle.ts),
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _require_ascending(symbol: str, candles: Sequence[Candle]) -> None:
    for earlier, later in zip(candles, candles[1:], strict=False):
        if later.ts <= earlier.ts:
            raise SnapshotIntegrityError(
                f"Candles for '{symbol}' must be strictly ascending by timestamp; "
                f"{later.ts.isoformat()} follows {earlier.ts.isoformat()}",
                field="candles",
            )


def _source_metadata(
    source_label: str,
    *,
    granularity: str,
    lookback: int,
    as_of: datetime,
) -> str:
    return (
        f"{source_label}:granularity={granularity}:lookback={lookback}"
        f":as_of={as_of.isoformat()}"
    )
