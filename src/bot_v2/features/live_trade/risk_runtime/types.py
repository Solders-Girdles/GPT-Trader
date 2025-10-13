"""Common type aliases for runtime risk helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

AnyLogger = Any
LogEventFn = Callable[[str, Mapping[str, Any], str | None], None]


__all__ = ["AnyLogger", "LogEventFn"]
