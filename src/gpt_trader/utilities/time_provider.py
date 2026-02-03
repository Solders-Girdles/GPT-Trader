"""Time/clock abstractions for deterministic time handling.

This module intentionally supports two related concepts:
- Clock: simple wall-clock time with now() + time()
- TimeProvider: richer interface used by runtime guards/status reporting

Implementation notes
- FakeClock is used heavily in tests. It supports both:
  - FakeClock(<datetime>)  (clock-style)
  - FakeClock(start_time=<float>) / set_datetime(...) (time-provider style)
"""

from __future__ import annotations

import time as time_module
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from gpt_trader.utilities.datetime_helpers import normalize_to_utc, utc_now


@runtime_checkable
class Clock(Protocol):
    """Protocol for retrieving current time."""

    def now(self) -> datetime:
        """Return the current time (timezone-aware UTC)."""

    def time(self) -> float:
        """Return the current time as epoch seconds."""


@runtime_checkable
class TimeProvider(Protocol):
    """Protocol for retrieving current time values."""

    def now_utc(self) -> datetime:
        """Return the current UTC time as a timezone-aware datetime."""

    def time(self) -> float:
        """Return the current Unix timestamp (seconds since epoch)."""

    def monotonic(self) -> float:
        """Return a monotonic clock value for measuring durations."""


class SystemClock:
    """Clock backed by the system time sources."""

    def now(self) -> datetime:
        return utc_now()

    def now_utc(self) -> datetime:
        # Alias used by TimeProvider consumers.
        return utc_now()

    def time(self) -> float:
        return time_module.time()

    def monotonic(self) -> float:
        return time_module.monotonic()


class FakeClock:
    """Deterministic clock for tests that can be advanced or reset."""

    def __init__(
        self,
        initial: datetime | None = None,
        *,
        start_time: float | None = None,
        start_datetime: datetime | None = None,
        start_monotonic: float | None = None,
    ) -> None:
        if initial is not None and start_datetime is None:
            start_datetime = initial

        if start_datetime is not None:
            now = normalize_to_utc(start_datetime)
            ts = now.timestamp()
        elif start_time is not None:
            ts = float(start_time)
            now = datetime.fromtimestamp(ts, UTC)
        else:
            now = normalize_to_utc(utc_now())
            ts = now.timestamp()

        if start_monotonic is None:
            start_monotonic = float(ts)

        self._now = now
        self._monotonic = float(start_monotonic)

    # Clock API
    def now(self) -> datetime:
        return self._now

    def time(self) -> float:
        return self._now.timestamp()

    # TimeProvider API
    def now_utc(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    # Mutators
    def set_time(self, current: datetime | float) -> None:
        if isinstance(current, datetime):
            self._now = normalize_to_utc(current)
            self._monotonic = float(self._now.timestamp())
            return

        ts = float(current)
        self._now = datetime.fromtimestamp(ts, UTC)
        self._monotonic = float(ts)

    def set_datetime(self, value: datetime) -> None:
        self.set_time(value)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() requires a non-negative duration")
        delta = float(seconds)
        self._now = self._now + timedelta(seconds=delta)
        self._monotonic += delta


_default_clock = SystemClock()
_clock: TimeProvider = _default_clock


def get_clock() -> TimeProvider:
    """Return the current active clock implementation."""

    return _clock


def set_clock(clock: TimeProvider) -> None:
    """Override the active clock (useful for deterministic tests)."""

    global _clock
    _clock = clock


def reset_clock() -> None:
    """Reset the active clock to the system clock."""

    global _clock
    _clock = _default_clock


__all__ = [
    "Clock",
    "TimeProvider",
    "SystemClock",
    "FakeClock",
    "get_clock",
    "set_clock",
    "reset_clock",
]
