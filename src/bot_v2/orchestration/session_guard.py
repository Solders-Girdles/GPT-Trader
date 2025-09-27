"""Trading session window utilities."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, time


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
