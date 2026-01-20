"""Tests for SimulationClock delay behavior based on clock speed."""

from __future__ import annotations

from datetime import timedelta

import pytest

import gpt_trader.backtesting.engine.clock as clock_module
from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed


@pytest.fixture()
def sleep_calls(monkeypatch) -> list[float]:
    calls: list[float] = []

    def _sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(clock_module.time, "sleep", _sleep)
    return calls


@pytest.fixture()
def async_sleep_calls(monkeypatch) -> list[float]:
    calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(clock_module.asyncio, "sleep", _sleep)
    return calls


class TestSimulationClockDelays:
    """Tests for delay application based on clock speed."""

    @pytest.mark.parametrize(
        ("speed", "delta", "expected_calls"),
        [
            (ClockSpeed.INSTANT, timedelta(minutes=5), []),
            (ClockSpeed.REAL_TIME, timedelta(seconds=1), [1.0]),
            (ClockSpeed.FAST_10X, timedelta(seconds=10), [1.0]),
            (ClockSpeed.FAST_100X, timedelta(seconds=100), [1.0]),
        ],
    )
    def test_delay_by_speed(
        self,
        sleep_calls: list[float],
        speed: ClockSpeed,
        delta: timedelta,
        expected_calls: list[float],
    ) -> None:
        clock = SimulationClock(speed=speed)

        clock.advance(delta)

        assert sleep_calls == expected_calls


class TestSimulationClockAsyncDelays:
    """Tests for async delay application."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("speed", "delta", "expected_calls"),
        [
            (ClockSpeed.INSTANT, timedelta(minutes=5), []),
            (ClockSpeed.REAL_TIME, timedelta(seconds=1), [1.0]),
            (ClockSpeed.FAST_10X, timedelta(seconds=10), [1.0]),
            (ClockSpeed.FAST_100X, timedelta(seconds=100), [1.0]),
        ],
    )
    async def test_delay_by_speed_async(
        self,
        async_sleep_calls: list[float],
        speed: ClockSpeed,
        delta: timedelta,
        expected_calls: list[float],
    ) -> None:
        clock = SimulationClock(speed=speed)

        await clock.advance_async(delta)

        assert async_sleep_calls == expected_calls
