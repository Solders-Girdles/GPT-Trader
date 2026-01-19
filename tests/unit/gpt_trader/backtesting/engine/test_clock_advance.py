"""Tests for SimulationClock advance and callbacks."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed


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
