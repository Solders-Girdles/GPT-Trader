"""Tests for SimulationClock delay behavior based on clock speed."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed


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
