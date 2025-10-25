"""Network and infrastructure logging mixin."""

from __future__ import annotations

from typing import Any

from .levels import LogLevel


class NetworkLoggingMixin:
    """Provide network and infrastructure logging helpers."""

    def log_market_heartbeat(
        self,
        source: str,
        last_update_ts: str,
        latency_ms: float | None = None,
        staleness_ms: float | None = None,
        threshold_ms: int | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="market_heartbeat",
            message=f"market heartbeat {source}",
            source=source,
            last_update_ts=last_update_ts,
            latency_ms=latency_ms,
            staleness_ms=staleness_ms,
            staleness_threshold_ms=threshold_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_ws_latency(self, stream: str, latency_ms: float, **kwargs: Any) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="ws_latency",
            message=f"ws {stream} latency={latency_ms:.2f}ms",
            stream=stream,
            latency_ms=latency_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_rest_response(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if 200 <= status_code < 400 else LogLevel.WARNING
        entry = self._create_log_entry(
            level=level,
            event_type="rest_timing",
            message=f"{method.upper()} {endpoint} {status_code} in {duration_ms:.1f}ms",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )
        self._emit_log(entry)


__all__ = ["NetworkLoggingMixin"]
