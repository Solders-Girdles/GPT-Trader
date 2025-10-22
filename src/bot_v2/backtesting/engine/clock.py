"""Simulation clock for controlling backtest replay speed."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Callable

from bot_v2.backtesting.types import ClockSpeed


class SimulationClock:
    """
    Controls the passage of time in backtesting simulations.

    The clock can run at different speeds:
    - INSTANT: As fast as possible (no delays)
    - REAL_TIME: 1:1 with wall clock time
    - FAST_10X: 10x faster than real-time
    - FAST_100X: 100x faster than real-time
    """

    def __init__(
        self,
        speed: ClockSpeed = ClockSpeed.INSTANT,
        start_time: datetime | None = None,
    ):
        """
        Initialize simulation clock.

        Args:
            speed: Clock speed mode
            start_time: Starting simulation time (defaults to now)
        """
        self.speed = speed
        self._sim_time = start_time or datetime.utcnow()
        self._wall_time_start = time.time()
        self._sim_time_start = self._sim_time

        # Callbacks
        self._on_tick_callbacks: list[Callable[[datetime], None]] = []

    def advance(self, delta: timedelta) -> datetime:
        """
        Advance simulation time by a delta.

        Args:
            delta: Time delta to advance (e.g., timedelta(minutes=5))

        Returns:
            New simulation time
        """
        self._sim_time += delta

        # Apply delay based on clock speed
        if self.speed != ClockSpeed.INSTANT:
            self._apply_delay(delta)

        # Trigger callbacks
        for callback in self._on_tick_callbacks:
            callback(self._sim_time)

        return self._sim_time

    async def advance_async(self, delta: timedelta) -> datetime:
        """
        Advance simulation time asynchronously.

        This is useful for async event loops that need to yield control.

        Args:
            delta: Time delta to advance

        Returns:
            New simulation time
        """
        self._sim_time += delta

        # Apply delay based on clock speed (async version)
        if self.speed != ClockSpeed.INSTANT:
            await self._apply_delay_async(delta)

        # Trigger callbacks
        for callback in self._on_tick_callbacks:
            callback(self._sim_time)

        return self._sim_time

    def set_time(self, new_time: datetime) -> None:
        """Set simulation time to a specific value."""
        self._sim_time = new_time

    def now(self) -> datetime:
        """Get current simulation time."""
        return self._sim_time

    def elapsed_sim_time(self) -> timedelta:
        """Get elapsed simulation time since start."""
        return self._sim_time - self._sim_time_start

    def elapsed_wall_time(self) -> float:
        """Get elapsed wall clock time since start (seconds)."""
        return time.time() - self._wall_time_start

    def speedup_ratio(self) -> float:
        """Calculate actual speedup ratio (sim_time / wall_time)."""
        wall_elapsed = self.elapsed_wall_time()
        if wall_elapsed == 0:
            return float("inf")

        sim_elapsed = self.elapsed_sim_time().total_seconds()
        return sim_elapsed / wall_elapsed

    def on_tick(self, callback: Callable[[datetime], None]) -> None:
        """Register a callback to be called on each clock tick."""
        self._on_tick_callbacks.append(callback)

    def _apply_delay(self, delta: timedelta) -> None:
        """Apply appropriate delay based on clock speed."""
        if self.speed == ClockSpeed.REAL_TIME:
            # 1:1 with wall clock
            time.sleep(delta.total_seconds())
        elif self.speed == ClockSpeed.FAST_10X:
            # 10x faster
            time.sleep(delta.total_seconds() / 10)
        elif self.speed == ClockSpeed.FAST_100X:
            # 100x faster
            time.sleep(delta.total_seconds() / 100)
        # INSTANT: no delay

    async def _apply_delay_async(self, delta: timedelta) -> None:
        """Apply appropriate delay asynchronously."""
        if self.speed == ClockSpeed.REAL_TIME:
            await asyncio.sleep(delta.total_seconds())
        elif self.speed == ClockSpeed.FAST_10X:
            await asyncio.sleep(delta.total_seconds() / 10)
        elif self.speed == ClockSpeed.FAST_100X:
            await asyncio.sleep(delta.total_seconds() / 100)
        # INSTANT: no delay
