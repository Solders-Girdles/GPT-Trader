"""Deterministic backoff policy helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackoffDecision:
    """Result of evaluating a backoff attempt."""

    attempt: int
    delay_seconds: float
    capped: bool


def evaluate_backoff_delay(
    *,
    attempt: int,
    base_delay: float,
    max_delay: float,
    multiplier: float = 2.0,
    jitter: float = 0.0,
    random_fn: Callable[[], float] | None = None,
) -> BackoffDecision:
    """Pure evaluation of a retry backoff delay without reading clocks."""

    if attempt <= 1 or base_delay <= 0:
        return BackoffDecision(attempt=attempt, delay_seconds=0.0, capped=False)

    raw_delay = base_delay * (multiplier ** (attempt - 2))
    capped = raw_delay >= max_delay
    delay = min(raw_delay, max_delay)

    if jitter > 0 and delay > 0:
        rng = random_fn or random.random
        delay += delay * jitter * rng()

    return BackoffDecision(attempt=attempt, delay_seconds=float(delay), capped=capped)


__all__ = [
    "BackoffDecision",
    "evaluate_backoff_delay",
]
