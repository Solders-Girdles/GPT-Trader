"""Edge coverage for CoinbaseHistoricalFetcher chunking and deduplication."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.backtesting.data.fetcher import CoinbaseHistoricalFetcher
from gpt_trader.core import Candle


@pytest.mark.asyncio
async def test_fetch_candles_deduplicates_and_sorts_chunks() -> None:
    client = Mock()
    fetcher = CoinbaseHistoricalFetcher(client=client)

    start = datetime(2024, 1, 1, 0, 0, 0)
    mid = datetime(2024, 1, 1, 0, 2, 0)
    end = datetime(2024, 1, 1, 0, 4, 0)

    candles_chunk_1 = [
        Candle(
            ts=start,
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            volume=Decimal("1"),
        ),
        Candle(
            ts=start + timedelta(minutes=1),
            open=Decimal("2"),
            high=Decimal("2"),
            low=Decimal("2"),
            close=Decimal("2"),
            volume=Decimal("2"),
        ),
    ]
    candles_chunk_2 = [
        Candle(
            ts=start + timedelta(minutes=1),
            open=Decimal("2"),
            high=Decimal("2"),
            low=Decimal("2"),
            close=Decimal("2"),
            volume=Decimal("2"),
        ),
        Candle(
            ts=start + timedelta(minutes=2),
            open=Decimal("3"),
            high=Decimal("3"),
            low=Decimal("3"),
            close=Decimal("3"),
            volume=Decimal("3"),
        ),
    ]

    fetcher._create_chunks = Mock(return_value=[(start, mid), (mid, end)])
    fetcher._fetch_chunk = AsyncMock(side_effect=[candles_chunk_1, candles_chunk_2])
    fetcher._rate_limit = AsyncMock()

    result = await fetcher.fetch_candles(
        symbol="BTC-USD",
        granularity="ONE_MINUTE",
        start=start,
        end=end,
    )

    assert [c.ts for c in result] == [
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=2),
    ]
    fetcher._fetch_chunk.assert_any_call(
        symbol="BTC-USD",
        granularity="ONE_MINUTE",
        start=start,
        end=mid,
    )
    fetcher._fetch_chunk.assert_any_call(
        symbol="BTC-USD",
        granularity="ONE_MINUTE",
        start=mid,
        end=end,
    )


def test_create_chunks_splits_over_limit() -> None:
    fetcher = CoinbaseHistoricalFetcher(client=Mock())
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=60 * 301)

    chunks = fetcher._create_chunks(start, end, candle_seconds=60, max_candles=300)

    assert len(chunks) == 2
    assert chunks[0] == (start, start + timedelta(seconds=60 * 300))
    assert chunks[1] == (start + timedelta(seconds=60 * 300), end)


def test_granularity_to_seconds_unknown_defaults() -> None:
    fetcher = CoinbaseHistoricalFetcher(client=Mock())

    assert fetcher._granularity_to_seconds("UNKNOWN") == 60
