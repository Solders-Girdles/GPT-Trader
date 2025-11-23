"""Trading session window utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any


@dataclass
class SessionStatus:
    is_trading_allowed: bool
    in_window: bool
    window_start: Any
    window_end: Any
    trading_days: list[str]
    next_change: str


class TradingSessionGuard:
    """Encapsulates trading window enforcement to keep orchestration focused."""

    def __init__(
        self,
        start: time | None,
        end: time | None,
        trading_days: list[str] | None,
    ) -> None:
        self._start = start
        self._end = end
        self._days = [d.lower() for d in (trading_days or [])]
        self._now: Callable[[], datetime] = datetime.now

    def is_trading_allowed(self, now: datetime | None = None) -> bool:
        """Alias for should_trade for compatibility."""
        return self.should_trade(now)

    def should_trade(self, now: datetime | None = None) -> bool:
        """Return True when the configured trading window allows execution."""
        if not self._start or not self._end or not self._days:
            return True

        current = now or self._now()
        try:
            day_name = current.strftime("%A").lower()
        except Exception:
            return True

        if day_name not in self._days:
            return False

        current_time = current.time()
        if self._start <= self._end:
            return self._start <= current_time <= self._end
        # Handle overnight windows (e.g., 22:00 -> 06:00)
        return current_time >= self._start or current_time <= self._end

    def set_clock(self, clock: Callable[[], datetime]) -> None:
        """Inject deterministic clock for testing."""
        self._now = clock

    def get_session_status(self) -> SessionStatus:
        """Get detailed session status."""
        now = self._now()
        in_window = self.should_trade(now)

        # Determine next window change (simplified)
        next_change = "unknown"

        return SessionStatus(
            is_trading_allowed=in_window,
            in_window=in_window,
            window_start=self._start,
            window_end=self._end,
            trading_days=self._days,
            next_change=next_change,
        )
