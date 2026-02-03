"""Clock abstractions for deterministic time handling."""

from __future__ import annotations

import time as system_time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


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

    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def time(self) -> float:
        return system_time.time()

    def monotonic(self) -> float:
        return system_time.monotonic()


def _coerce_to_utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    elif value.tzinfo != UTC:
        value = value.astimezone(UTC)
    return value.timestamp()


class FakeClock:
    """Deterministic clock for tests that can be advanced or reset."""

    def __init__(
        self,
        *,
        start_time: float | None = None,
        start_datetime: datetime | None = None,
        start_monotonic: float | None = None,
    ) -> None:
        if start_datetime is not None:
            start_time = _coerce_to_utc_timestamp(start_datetime)
        if start_time is None:
            start_time = 0.0
        if start_monotonic is None:
            start_monotonic = float(start_time)
        self._time = float(start_time)
        self._monotonic = float(start_monotonic)

    def now_utc(self) -> datetime:
        return datetime.fromtimestamp(self._time, UTC)

    def time(self) -> float:
        return self._time

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() requires a non-negative duration")
        delta = float(seconds)
        self._time += delta
        self._monotonic += delta

    def set_time(self, timestamp: float) -> None:
        self._time = float(timestamp)
        self._monotonic = float(timestamp)

    def set_datetime(self, value: datetime) -> None:
        self.set_time(_coerce_to_utc_timestamp(value))


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
    "TimeProvider",
    "SystemClock",
    "FakeClock",
    "get_clock",
    "set_clock",
    "reset_clock",
]
