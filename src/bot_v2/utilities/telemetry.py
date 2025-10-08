"""Telemetry and metrics emission utilities.

Provides centralized helpers for emitting metrics to the event store with
consistent error handling and logging fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.monitoring.system.logger import ProductionLogger
    from bot_v2.persistence.event_store import EventStore


def emit_metric(
    event_store: EventStore | None,
    bot_id: str,
    payload: dict[str, Any],
    *,
    logger: ProductionLogger | None = None,
) -> None:
    """Safely emit a metric to the event store with fallback logging on failure.

    This centralizes the repetitive try/except pattern used throughout the codebase
    when emitting metrics. If the event store fails, the error is logged to the
    production logger (if provided) instead of raising an exception.

    Args:
        event_store: Event store to write metrics to (may be None)
        bot_id: Bot identifier for the metric
        payload: Metric data dictionary
        logger: Optional production logger for fallback logging on failure

    Example:
        >>> from bot_v2.persistence.event_store import EventStore
        >>> from bot_v2.monitoring.system import get_logger
        >>> store = EventStore()
        >>> logger = get_logger()
        >>> emit_metric(
        ...     store,
        ...     "perps_bot",
        ...     {"event_type": "ws_mark_update", "symbol": "BTC-PERP"},
        ...     logger=logger
        ... )
    """
    if event_store is None:
        return

    try:
        event_store.append_metric(bot_id, payload)
    except Exception as exc:
        # Metric emission is best-effort; log failure but don't raise
        if logger is not None:
            from bot_v2.monitoring.system.logger import LogLevel

            logger.log_event(
                LogLevel.DEBUG,
                "metric_emit_failed",
                "Failed to emit metric to event store",
                error=str(exc),
                bot_id=bot_id,
                metric_type=payload.get("event_type", "unknown"),
            )


__all__ = ["emit_metric"]
