from __future__ import annotations

from collections.abc import Iterator

import pytest

from gpt_trader.security import rate_limiter as rate_limiter_module


class RateLimiterClock:
    """Controllable stand-in for the rate limiter's ``time`` module.

    The :class:`~gpt_trader.security.rate_limiter.RateLimiter` reads wall-clock
    time via the module-level ``time`` reference. Swapping that reference for
    this object lets tests advance time deterministically without freezegun's
    global monkeypatching (which also patches ``perf_counter`` and pollutes
    pytest's ``--durations`` reporting).
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


@pytest.fixture
def rate_limiter_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[RateLimiterClock]:
    """Replace the rate limiter's wall clock with a deterministic fake."""
    clock = RateLimiterClock()
    monkeypatch.setattr(rate_limiter_module, "time", clock)
    yield clock
