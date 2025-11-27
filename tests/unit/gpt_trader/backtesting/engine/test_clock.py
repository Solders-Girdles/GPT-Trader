"""Tests for simulation clock module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed


class TestSimulationClockInit:
    """Tests for SimulationClock initialization."""

    def test_default_speed_is_instant(self) -> None:
        clock = SimulationClock()
        assert clock.speed == ClockSpeed.INSTANT

    def test_custom_speed(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_10X)
        assert clock.speed == ClockSpeed.FAST_10X

    def test_default_start_time_is_now(self) -> None:
        before = datetime.utcnow()
        clock = SimulationClock()
        after = datetime.utcnow()
        assert before <= clock.now() <= after

    def test_custom_start_time(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(start_time=start)
        assert clock.now() == start

    def test_initializes_empty_callbacks(self) -> None:
        clock = SimulationClock()
        assert clock._on_tick_callbacks == []


class TestSimulationClockAdvance:
    """Tests for clock advance functionality."""

    def test_advance_updates_time(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        result = clock.advance(timedelta(minutes=5))

        expected = start + timedelta(minutes=5)
        assert result == expected
        assert clock.now() == expected

    def test_advance_triggers_callbacks(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        callback = MagicMock()
        clock.on_tick(callback)

        clock.advance(timedelta(minutes=5))

        callback.assert_called_once_with(start + timedelta(minutes=5))

    def test_advance_triggers_multiple_callbacks(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        callback1 = MagicMock()
        callback2 = MagicMock()
        clock.on_tick(callback1)
        clock.on_tick(callback2)

        clock.advance(timedelta(minutes=5))

        expected_time = start + timedelta(minutes=5)
        callback1.assert_called_once_with(expected_time)
        callback2.assert_called_once_with(expected_time)

    def test_multiple_advances_accumulate(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        clock.advance(timedelta(minutes=5))
        clock.advance(timedelta(minutes=10))
        clock.advance(timedelta(hours=1))

        expected = start + timedelta(minutes=5) + timedelta(minutes=10) + timedelta(hours=1)
        assert clock.now() == expected


class TestSimulationClockAdvanceAsync:
    """Tests for async clock advance functionality."""

    @pytest.mark.asyncio
    async def test_advance_async_updates_time(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        result = await clock.advance_async(timedelta(minutes=5))

        expected = start + timedelta(minutes=5)
        assert result == expected
        assert clock.now() == expected

    @pytest.mark.asyncio
    async def test_advance_async_triggers_callbacks(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        clock = SimulationClock(speed=ClockSpeed.INSTANT, start_time=start)

        callback = MagicMock()
        clock.on_tick(callback)

        await clock.advance_async(timedelta(minutes=5))

        callback.assert_called_once_with(start + timedelta(minutes=5))


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


class TestSimulationClockOnTick:
    """Tests for on_tick callback registration."""

    def test_on_tick_adds_callback(self) -> None:
        clock = SimulationClock()
        callback = MagicMock()

        clock.on_tick(callback)

        assert callback in clock._on_tick_callbacks

    def test_multiple_callbacks_can_be_registered(self) -> None:
        clock = SimulationClock()
        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()

        clock.on_tick(cb1)
        clock.on_tick(cb2)
        clock.on_tick(cb3)

        assert len(clock._on_tick_callbacks) == 3
        assert cb1 in clock._on_tick_callbacks
        assert cb2 in clock._on_tick_callbacks
        assert cb3 in clock._on_tick_callbacks


class TestSimulationClockDelays:
    """Tests for delay application based on clock speed."""

    def test_instant_speed_no_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.INSTANT)

        with patch("time.sleep") as mock_sleep:
            clock.advance(timedelta(minutes=5))
            mock_sleep.assert_not_called()

    def test_real_time_speed_full_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.REAL_TIME)

        with patch("time.sleep") as mock_sleep:
            clock.advance(timedelta(seconds=1))
            mock_sleep.assert_called_once_with(1.0)

    def test_fast_10x_speed_reduced_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_10X)

        with patch("time.sleep") as mock_sleep:
            clock.advance(timedelta(seconds=10))
            mock_sleep.assert_called_once_with(1.0)  # 10s / 10 = 1s

    def test_fast_100x_speed_minimal_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_100X)

        with patch("time.sleep") as mock_sleep:
            clock.advance(timedelta(seconds=100))
            mock_sleep.assert_called_once_with(1.0)  # 100s / 100 = 1s


class TestSimulationClockAsyncDelays:
    """Tests for async delay application."""

    @pytest.mark.asyncio
    async def test_instant_speed_no_async_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.INSTANT)

        with patch("asyncio.sleep") as mock_sleep:
            await clock.advance_async(timedelta(minutes=5))
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_time_speed_async_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.REAL_TIME)

        with patch("asyncio.sleep") as mock_sleep:
            await clock.advance_async(timedelta(seconds=1))
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_fast_10x_speed_async_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_10X)

        with patch("asyncio.sleep") as mock_sleep:
            await clock.advance_async(timedelta(seconds=10))
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_fast_100x_speed_async_delay(self) -> None:
        clock = SimulationClock(speed=ClockSpeed.FAST_100X)

        with patch("asyncio.sleep") as mock_sleep:
            await clock.advance_async(timedelta(seconds=100))
            mock_sleep.assert_called_once_with(1.0)
