"""Tests for ClockedBarRunner initialization."""

from __future__ import annotations

from datetime import datetime

from tests.unit.gpt_trader.backtesting.engine.bar_runner_test_utils import (  # naming: allow
    _create_mock_data_provider,
)

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner
from gpt_trader.backtesting.types import ClockSpeed


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
