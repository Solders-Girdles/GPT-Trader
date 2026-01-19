"""Tests for SimulationClock initialization."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed
from gpt_trader.utilities.datetime_helpers import utc_now


class TestSimulationClockInit:
    """Tests for SimulationClock initialization."""

    def test_default_speed_is_instant(self) -> None:
        clock = SimulationClock()
        assert clock.speed == ClockSpeed.INSTANT

    def test_custom_speed(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_10X)
        assert clock.speed == ClockSpeed.FAST_10X

    def test_default_start_time_is_now(self) -> None:
        before = utc_now()
        clock = SimulationClock()
        after = utc_now()
        assert before <= clock.now() <= after

    def test_custom_start_time(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(start_time=start)
        assert clock.now() == start

    def test_initializes_empty_callbacks(self) -> None:
        clock = SimulationClock()
        assert clock._on_tick_callbacks == []
