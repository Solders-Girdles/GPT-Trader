"""Tests for bar runner module."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner, IHistoricalDataProvider
from gpt_trader.backtesting.types import ClockSpeed
from gpt_trader.features.brokerages.core.interfaces import Candle


def _create_mock_candle(
    symbol: str = "BTC-USD",
    ts: datetime | None = None,
    open_: Decimal = Decimal("50000"),
    high: Decimal = Decimal("50500"),
    low: Decimal = Decimal("49500"),
    close: Decimal = Decimal("50100"),
    volume: Decimal = Decimal("100"),
) -> Candle:
    """Create a mock candle for testing."""
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
    """Create a mock data provider."""
    provider = MagicMock(spec=IHistoricalDataProvider)

    async def mock_get_candles(
        symbol: str, granularity: str, start: datetime, end: datetime
    ) -> list[Candle]:
        if candles_by_symbol and symbol in candles_by_symbol:
            # Return candles within the time range
            return [c for c in candles_by_symbol[symbol] if start <= c.ts < end]
        return []

    provider.get_candles = AsyncMock(side_effect=mock_get_candles)
    return provider


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


class TestParseGranularity:
    """Tests for granularity parsing."""

    def test_one_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=1)

    def test_five_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=5)

    def test_fifteen_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="FIFTEEN_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=15)

    def test_one_hour(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(hours=1)

    def test_one_day(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_DAY",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        assert runner._granularity_delta == timedelta(days=1)

    def test_unsupported_granularity_raises_error(self) -> None:
        provider = _create_mock_data_provider()
        with pytest.raises(ValueError, match="Unsupported granularity"):
            ClockedBarRunner(
                data_provider=provider,
                symbols=["BTC-USD"],
                granularity="INVALID",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )


class TestBarsToQuotes:
    """Tests for bar to quote conversion."""

    def test_converts_bar_to_quote(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("50000"))
        bar_time = datetime(2024, 1, 1, 12, 0, 0)

        quotes = runner._bars_to_quotes({"BTC-USD": bar}, bar_time)

        assert "BTC-USD" in quotes
        quote = quotes["BTC-USD"]
        assert quote.symbol == "BTC-USD"
        assert quote.last == Decimal("50000")
        assert quote.ts == bar_time
        # Bid should be slightly below close
        assert quote.bid < quote.last
        # Ask should be slightly above close
        assert quote.ask > quote.last

    def test_spread_calculation(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("10000"))
        quotes = runner._bars_to_quotes({"BTC-USD": bar}, datetime(2024, 1, 1))

        quote = quotes["BTC-USD"]
        # Spread should be ~0.1% (5 bps on each side)
        spread = quote.ask - quote.bid
        expected_spread = Decimal("10000") * Decimal("10") / Decimal("20000")  # ~5
        assert abs(spread - expected_spread) < Decimal("0.01")

    def test_multiple_symbols(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD", "ETH-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bars = {
            "BTC-USD": _create_mock_candle(symbol="BTC-USD", close=Decimal("50000")),
            "ETH-USD": _create_mock_candle(symbol="ETH-USD", close=Decimal("3000")),
        }
        quotes = runner._bars_to_quotes(bars, datetime(2024, 1, 1))

        assert len(quotes) == 2
        assert "BTC-USD" in quotes
        assert "ETH-USD" in quotes
        assert quotes["BTC-USD"].last == Decimal("50000")
        assert quotes["ETH-USD"].last == Decimal("3000")


class TestBarRunnerHooks:
    """Tests for event hooks."""

    def test_on_bar_start_adds_callback(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        callback = MagicMock()
        runner.on_bar_start(callback)

        assert callback in runner._on_bar_start_hooks

    def test_on_bar_end_adds_callback(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        callback = MagicMock()
        runner.on_bar_end(callback)

        assert callback in runner._on_bar_end_hooks


class TestBarRunnerProgress:
    """Tests for progress tracking."""

    def test_progress_starts_at_zero(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert runner.progress_pct == pytest.approx(0.0, abs=0.1)

    def test_progress_100_when_duration_zero(self) -> None:
        provider = _create_mock_data_provider()
        start = datetime(2024, 1, 1, 12, 0, 0)
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start,  # Same as start
        )

        assert runner.progress_pct == 100.0

    def test_bars_remaining_initial(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 10, 0, 0),  # 10 hours
        )

        # Should have ~10 bars remaining
        assert runner.bars_remaining == 10

    def test_bars_remaining_zero_duration(self) -> None:
        provider = _create_mock_data_provider()
        start = datetime(2024, 1, 1, 12, 0, 0)
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start,
        )

        assert runner.bars_remaining == 0


class TestBarRunnerRun:
    """Tests for the async run generator."""

    @pytest.mark.asyncio
    async def test_run_yields_bars_and_quotes(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candle1 = _create_mock_candle(symbol="BTC-USD", ts=start)
        candle2 = _create_mock_candle(symbol="BTC-USD", ts=start + timedelta(hours=1))

        provider = _create_mock_data_provider(
            {
                "BTC-USD": [candle1, candle2],
            }
        )

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=2),
        )

        results = []
        async for bar_time, bars, quotes in runner.run():
            results.append((bar_time, bars, quotes))

        assert len(results) == 2
        assert results[0][0] == start
        assert "BTC-USD" in results[0][1]
        assert "BTC-USD" in results[0][2]

    @pytest.mark.asyncio
    async def test_run_skips_empty_bars(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        # Only provide data for hour 0, skip hour 1
        candle1 = _create_mock_candle(symbol="BTC-USD", ts=start)

        provider = _create_mock_data_provider(
            {
                "BTC-USD": [candle1],
            }
        )

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=3),
        )

        results = []
        async for bar_time, bars, quotes in runner.run():
            results.append((bar_time, bars, quotes))

        # Should only yield 1 bar (hour 0), skipping hours 1 and 2
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_triggers_hooks(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candle = _create_mock_candle(symbol="BTC-USD", ts=start)

        provider = _create_mock_data_provider({"BTC-USD": [candle]})

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=1),
        )

        bar_start_hook = MagicMock()
        bar_end_hook = MagicMock()
        runner.on_bar_start(bar_start_hook)
        runner.on_bar_end(bar_end_hook)

        async for _ in runner.run():
            pass

        bar_start_hook.assert_called_once()
        bar_end_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_updates_bars_processed(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        candles = [
            _create_mock_candle(symbol="BTC-USD", ts=start + timedelta(hours=i)) for i in range(3)
        ]

        provider = _create_mock_data_provider({"BTC-USD": candles})

        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start + timedelta(hours=3),
        )

        assert runner._bars_processed == 0

        async for _ in runner.run():
            pass

        assert runner._bars_processed == 3


class TestIHistoricalDataProvider:
    """Tests for the IHistoricalDataProvider interface."""

    def test_interface_cannot_be_instantiated(self) -> None:
        """Verify that IHistoricalDataProvider is an abstract base class."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IHistoricalDataProvider()  # type: ignore[abstract]
