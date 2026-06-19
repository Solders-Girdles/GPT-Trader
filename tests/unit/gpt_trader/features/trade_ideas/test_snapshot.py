from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.core import Candle
from gpt_trader.features.trade_ideas import (
    MarketSnapshot,
    SnapshotIntegrityError,
    SymbolSeries,
)

AS_OF = datetime(2026, 6, 12, 0, 0, tzinfo=UTC)


def make_candle(days_before_as_of: int, close: str = "100") -> Candle:
    price = Decimal(close)
    return Candle(
        ts=AS_OF - timedelta(days=days_before_as_of),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1000"),
    )


def test_valid_snapshot_construction() -> None:
    series = SymbolSeries(
        symbol="BTC-USD",
        granularity="1d",
        candles=(make_candle(3), make_candle(2), make_candle(1)),
    )

    snapshot = MarketSnapshot(as_of=AS_OF, source="coinbase:candles", series=(series,))

    assert snapshot.symbols() == ("BTC-USD",)
    assert snapshot.series_for("BTC-USD") is series
    assert snapshot.series_for("ETH-USD") is None


def test_candle_at_or_after_as_of_is_rejected() -> None:
    series = SymbolSeries(
        symbol="BTC-USD",
        granularity="1d",
        candles=(make_candle(1), make_candle(0)),
    )

    with pytest.raises(SnapshotIntegrityError, match="look-ahead"):
        MarketSnapshot(as_of=AS_OF, source="coinbase:candles", series=(series,))


def test_unordered_candles_are_rejected() -> None:
    with pytest.raises(SnapshotIntegrityError, match="ascending"):
        SymbolSeries(
            symbol="BTC-USD",
            granularity="1d",
            candles=(make_candle(1), make_candle(2)),
        )


def test_duplicate_symbols_are_rejected() -> None:
    series = SymbolSeries(symbol="BTC-USD", granularity="1d", candles=(make_candle(1),))

    with pytest.raises(SnapshotIntegrityError, match="Duplicate"):
        MarketSnapshot(as_of=AS_OF, source="coinbase:candles", series=(series, series))


def test_empty_series_has_no_last_close() -> None:
    series = SymbolSeries(symbol="BTC-USD", granularity="1d", candles=())

    with pytest.raises(SnapshotIntegrityError, match="no candles"):
        series.last_close()
