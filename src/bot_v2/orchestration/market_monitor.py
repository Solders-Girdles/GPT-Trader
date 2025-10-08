"""Minimal market activity monitor used by the orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta, timezone

from bot_v2.utilities import to_iso_utc, utc_now


class MarketActivityMonitor:
    """Track market data freshness and transport health."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        max_failures: int = 3,
        max_staleness_ms: int = 5_000,
        heartbeat_logger: Callable[..., None] | None = None,
    ) -> None:
        self.last_update = {sym: utc_now() for sym in symbols}
        self.consecutive_failures: int = 0
        self.max_failures = max_failures
        self.max_staleness_ms = max_staleness_ms
        self._heartbeat_logger = heartbeat_logger

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_update(self, symbol: str) -> None:
        now = utc_now()
        self.last_update[symbol] = now
        if self._heartbeat_logger is not None:
            try:
                self._heartbeat_logger(
                    source="rest_quote",
                    last_update_ts=to_iso_utc(now),
                )
            except Exception:
                # Heartbeat logging is best effort
                pass

    def check_staleness(self) -> bool:
        if not self.last_update:
            return False
        threshold = timedelta(milliseconds=self.max_staleness_ms)
        now = utc_now()
        for symbol, ts in self.last_update.items():
            last_ts = ts
            if last_ts.tzinfo is None and now.tzinfo is not None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            if now - last_ts > threshold:
                if self._heartbeat_logger is not None:
                    try:
                        self._heartbeat_logger(
                            source="staleness_guard",
                            symbol=symbol,
                            last_update_ts=to_iso_utc(last_ts),
                            staleness_ms=(now - last_ts).total_seconds() * 1000.0,
                            threshold_ms=self.max_staleness_ms,
                        )
                    except Exception:
                        pass
                return True
        return False

    def should_fallback_to_rest(self) -> bool:
        return self.consecutive_failures > self.max_failures or self.check_staleness()
