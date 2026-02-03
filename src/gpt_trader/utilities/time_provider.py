"""Time provider abstractions for deterministic testing."""

from __future__ import annotations

import time as time_module
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from gpt_trader.utilities.datetime_helpers import normalize_to_utc, utc_now


@runtime_checkable
class Clock(Protocol):
    """Protocol for retrieving current time."""

    def now(self) -> datetime:
        """Return the current time."""

    def time(self) -> float:
        """Return the current time as epoch seconds."""


class SystemClock:
    """Clock backed by the system's UTC time."""

    def now(self) -> datetime:
        return utc_now()

    def time(self) -> float:
        return time_module.time()


class FakeClock:
    """Deterministic clock for tests."""

    def __init__(self, initial: datetime | None = None) -> None:
        base_time = initial or utc_now()
        self._now = normalize_to_utc(base_time)

    def now(self) -> datetime:
        return self._now

    def time(self) -> float:
        return self._now.timestamp()

    def set_time(self, current: datetime) -> None:
        self._now = normalize_to_utc(current)

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


@runtime_checkable
class TimeProvider(Protocol):
    """Protocol for retrieving current time."""

    def now(self) -> datetime:
        """Return the current time."""


class SystemTimeProvider:
    """Time provider backed by the system clock."""

    def now(self) -> datetime:
        return datetime.now()


__all__ = [
    "Clock",
    "SystemClock",
    "FakeClock",
    "TimeProvider",
    "SystemTimeProvider",
]
