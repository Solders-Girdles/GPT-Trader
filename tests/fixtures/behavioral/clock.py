"""Deterministic clock helpers for tests."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable


class FakeClock:
    """Simple monotonic clock that advances manually."""

    def __init__(self, start: float | None = None) -> None:
        # Default to real time so code that inspects absolute timestamps still sees plausible values.
        self._current = float(start if start is not None else time.time())
        self._yield: Callable[[], Awaitable[None]] | None = None
        self._sleep_yield: Callable[[], None] | None = None

    def __call__(self) -> float:
        """Allow the instance to act like time.time."""
        return self._current

    def now(self) -> float:
        """Return current fake timestamp."""
        return self._current

    def reset(self, value: float) -> None:
        self._current = float(value)

    def advance(self, seconds: float) -> float:
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError("FakeClock cannot go backwards")
        self._current += seconds
        return self._current

    def sleep(self, seconds: float = 0.0) -> None:
        """Synchronous sleep that just increments the clock."""
        self.advance(seconds)
        if self._sleep_yield is not None:
            self._sleep_yield()

    async def async_sleep(self, seconds: float = 0.0) -> None:
        """Async sleep counterpart; mirrors asyncio.sleep interface."""
        self.advance(seconds)
        if self._yield is not None:
            await self._yield()

    def set_async_yield(self, factory: Callable[[], Awaitable[None]]) -> None:
        """Install a coroutine factory used to yield control once per async sleep."""
        self._yield = factory

    def set_thread_yield(self, callback: Callable[[], None]) -> None:
        """Install a callback used to yield the GIL for threaded code paths."""
        self._sleep_yield = callback


__all__ = ["FakeClock"]
