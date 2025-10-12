"""Common type aliases for runtime risk helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

AnyLogger = Any
LogEventFn = Callable[[str, dict[str, str], str], None]


__all__ = ["AnyLogger", "LogEventFn"]
