"""Time provider abstractions for deterministic testing."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class TimeProvider(Protocol):
    """Protocol for retrieving current time."""

    def now(self) -> datetime:
        """Return the current time."""


class SystemTimeProvider:
    """Time provider backed by the system clock."""

    def now(self) -> datetime:
        return datetime.now()


__all__ = ["TimeProvider", "SystemTimeProvider"]
