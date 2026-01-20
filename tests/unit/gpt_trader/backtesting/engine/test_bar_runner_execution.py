"""Tests for ClockedBarRunner.run(), hooks, and progress tracking."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.engine.bar_runner_test_utils import (  # naming: allow
    _create_mock_candle,
    _create_mock_data_provider,
)

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner


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
            end_date=start,
        )

        assert runner.progress_pct == 100.0

    def test_bars_remaining_initial(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 10, 0, 0),
        )

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
