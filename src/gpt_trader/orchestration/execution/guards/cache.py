"""
Guard state cache for temporal caching of runtime guard state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.orchestration.execution.guards.protocol import RuntimeGuardState


class GuardStateCache:
    """Manages temporal caching of guard state to avoid frequent recollection."""

    def __init__(
        self,
        full_interval: float = 60.0,
        invalidate_callback: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize guard state cache.

        Args:
            full_interval: Seconds between full state collections
            invalidate_callback: Optional callback invoked on cache invalidation
        """
        self._state: RuntimeGuardState | None = None
        self._dirty: bool = True
        self._last_full_ts: float = 0.0
        self._full_interval: float = full_interval
        self._invalidate_callback = invalidate_callback

    @property
    def state(self) -> RuntimeGuardState | None:
        """Get cached state if available."""
        return self._state

    def is_valid_for_incremental(self, now: float) -> bool:
        """Check if cached state can be reused for incremental check."""
        if self._dirty:
            return False
        if self._state is None:
            return False
        return (now - self._last_full_ts) < self._full_interval

    def should_run_full(self, now: float) -> bool:
        """Check if a full guard run is needed."""
        return not self.is_valid_for_incremental(now)

    def update(self, state: RuntimeGuardState, now: float) -> None:
        """Update cached state after full collection."""
        self._state = state
        self._last_full_ts = now
        self._dirty = False

    def invalidate(self) -> None:
        """Force refresh on next check."""
        self._state = None
        self._dirty = True
        if self._invalidate_callback is not None:
            self._invalidate_callback()


__all__ = ["GuardStateCache"]
