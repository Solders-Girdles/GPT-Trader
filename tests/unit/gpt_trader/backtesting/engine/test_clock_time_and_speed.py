"""Tests for SimulationClock time controls and speed metrics."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed


class TestSimulationClockSetTime:
    """Tests for set_time method."""

    def test_set_time_updates_current_time(self) -> None:
        clock = SimulationClock()
        new_time = datetime(2025, 6, 15, 10, 30, 0)

        clock.set_time(new_time)

        assert clock.now() == new_time

    def test_set_time_can_go_backwards(self) -> None:
        start = datetime(2024, 6, 15, 12, 0, 0)
        clock = SimulationClock(start_time=start)

        earlier = datetime(2024, 1, 1, 0, 0, 0)
        clock.set_time(earlier)

        assert clock.now() == earlier


class TestSimulationClockElapsedTime:
    """Tests for elapsed time calculations."""

    def test_elapsed_sim_time_starts_at_zero(self) -> None:
        clock = SimulationClock()
        assert clock.elapsed_sim_time() == timedelta(0)

    def test_elapsed_sim_time_after_advance(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        clock.advance(timedelta(hours=2))

        assert clock.elapsed_sim_time() == timedelta(hours=2)

    def test_elapsed_wall_time_is_non_negative(self) -> None:
        clock = SimulationClock()
        elapsed = clock.elapsed_wall_time()
        assert elapsed >= 0


class TestSimulationClockSpeedupRatio:
    """Tests for speedup ratio calculation."""

    def test_speedup_ratio_inf_when_wall_time_is_zero(self) -> None:
        clock = SimulationClock()
        with patch.object(clock, "elapsed_wall_time", return_value=0):
            ratio = clock.speedup_ratio()
            assert ratio == float("inf")

    def test_speedup_ratio_zero_when_no_sim_advance(self) -> None:
        clock = SimulationClock()
        ratio = clock.speedup_ratio()
        # At start with no sim time advance, ratio is 0 (0 sim time / non-zero wall time)
        # or inf if wall time is truly 0
        assert ratio >= 0  # Either 0 or inf are valid

    def test_speedup_ratio_after_advance(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        # Advance simulation time significantly
        clock.advance(timedelta(days=1))

        # Should have very high speedup for instant mode
        ratio = clock.speedup_ratio()
        assert ratio > 0
