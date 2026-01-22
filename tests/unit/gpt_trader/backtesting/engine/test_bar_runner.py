"""Tests for ClockedBarRunner initialization and chaos behavior."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.engine.bar_runner_test_utils import (  # naming: allow
    _ChaosStub,
    _create_mock_candle,
    _create_mock_data_provider,
)

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner
from gpt_trader.backtesting.types import ClockSpeed


def _build_runner(
    *,
    granularity: str = "ONE_HOUR",
    symbols: list[str] | None = None,
) -> ClockedBarRunner:
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1)
    return ClockedBarRunner(
        data_provider=_create_mock_data_provider(),
        symbols=symbols or ["BTC-USD"],
        granularity=granularity,
        start_date=start,
        end_date=end,
    )


class TestClockedBarRunnerInit:
    """Tests for ClockedBarRunner initialization."""

    def test_basic_initialization(self) -> None:
        provider = _create_mock_data_provider()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=end,
        )

        assert runner.symbols == ["BTC-USD"]
        assert runner.granularity == "ONE_HOUR"
        assert runner.start_date == start
        assert runner.end_date == end

    def test_default_clock_speed_is_instant(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert runner.clock.speed == ClockSpeed.INSTANT

    def test_custom_clock_speed(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            clock_speed=ClockSpeed.FAST_10X,
        )

        assert runner.clock.speed == ClockSpeed.FAST_10X

    def test_multiple_symbols(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert len(runner.symbols) == 3
        assert "BTC-USD" in runner.symbols
        assert "ETH-USD" in runner.symbols
        assert "SOL-USD" in runner.symbols


class TestBarRunnerChaos:
    """Tests for chaos integration in ClockedBarRunner."""

    @pytest.mark.asyncio
    async def test_run_skips_bars_dropped_by_chaos(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candle = _create_mock_candle(symbol="BTC-USD", ts=start)
        provider = _create_mock_data_provider({"BTC-USD": [candle]})

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=1),
            chaos_engine=_ChaosStub(drop_candles=True),
        )

        results = []
        async for bar_time, bars, quotes in runner.run():
            results.append((bar_time, bars, quotes))

        assert results == []

    @pytest.mark.asyncio
    async def test_run_applies_latency(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candle = _create_mock_candle(symbol="BTC-USD", ts=start)
        provider = _create_mock_data_provider({"BTC-USD": [candle]})
        latency = timedelta(minutes=2)

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=1),
            chaos_engine=_ChaosStub(latency=latency),
        )

        results = []
        async for bar_time, bars, quotes in runner.run():
            results.append((bar_time, bars, quotes))

        assert len(results) == 1
        assert results[0][0] == start + latency

    @pytest.mark.asyncio
    async def test_quotes_use_chaos_adjusted_candle_timestamp(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candle = _create_mock_candle(symbol="BTC-USD", ts=start)
        provider = _create_mock_data_provider({"BTC-USD": [candle]})
        offset = timedelta(minutes=-1)

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=1),
            chaos_engine=_ChaosStub(ts_offset=offset),
        )

        async for _, _, quotes in runner.run():
            assert quotes["BTC-USD"].ts == start + offset
            break


class TestParseGranularity:
    """Tests for granularity parsing."""

    @pytest.mark.parametrize(
        ("granularity", "expected"),
        [
            ("ONE_MINUTE", timedelta(minutes=1)),
            ("FIVE_MINUTE", timedelta(minutes=5)),
            ("FIFTEEN_MINUTE", timedelta(minutes=15)),
            ("ONE_HOUR", timedelta(hours=1)),
            ("ONE_DAY", timedelta(days=1)),
        ],
    )
    def test_supported_granularity(self, granularity: str, expected: timedelta) -> None:
        runner = _build_runner(granularity=granularity)
        assert runner._granularity_delta == expected

    def test_unsupported_granularity_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported granularity"):
            _build_runner(granularity="INVALID")


class TestBarsToQuotes:
    """Tests for bar to quote conversion."""

    def test_converts_bar_to_quote(self) -> None:
        runner = _build_runner()
        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("50000"))
        bar_time = datetime(2024, 1, 1, 12, 0, 0)

        quotes = runner._bars_to_quotes({"BTC-USD": bar}, bar_time)

        quote = quotes["BTC-USD"]
        assert quote.symbol == "BTC-USD"
        assert quote.last == Decimal("50000")
        assert quote.ts == bar_time
        assert quote.bid < quote.last
        assert quote.ask > quote.last

    def test_spread_calculation(self) -> None:
        runner = _build_runner()
        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("10000"))
        quotes = runner._bars_to_quotes({"BTC-USD": bar}, datetime(2024, 1, 1))

        quote = quotes["BTC-USD"]
        spread = quote.ask - quote.bid
        expected_spread = Decimal("10000") * Decimal("10") / Decimal("20000")
        assert abs(spread - expected_spread) < Decimal("0.01")

    def test_multiple_symbols(self) -> None:
        runner = _build_runner(symbols=["BTC-USD", "ETH-USD"])
        bars = {
            "BTC-USD": _create_mock_candle(symbol="BTC-USD", close=Decimal("50000")),
            "ETH-USD": _create_mock_candle(symbol="ETH-USD", close=Decimal("3000")),
        }
        quotes = runner._bars_to_quotes(bars, datetime(2024, 1, 1))

        assert len(quotes) == 2
        assert quotes["BTC-USD"].last == Decimal("50000")
        assert quotes["ETH-USD"].last == Decimal("3000")
