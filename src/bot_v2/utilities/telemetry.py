"""Telemetry and metrics emission utilities.

Provides centralized helpers for emitting metrics to the event store with
consistent error handling and logging fallbacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.persistence.event_store import EventStore


def emit_metric(
    event_store: EventStore | None,
    bot_id: str,
    payload: Mapping[str, Any],
    *,
    logger: Any | None = None,
    raise_on_error: bool = False,
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
        raise_on_error: Re-raise exceptions when True (defaults to False)

    Example:
        >>> from bot_v2.persistence.event_store import EventStore
        >>> from bot_v2.monitoring.system import get_logger
        >>> store = EventStore()
        >>> logger = get_logger()
        >>> emit_metric(
        ...     store,
        ...     "coinbase_trader",
        ...     {"event_type": "ws_mark_update", "symbol": "BTC-PERP"},
        ...     logger=logger
        ... )
    """
    if event_store is None:
        return

    metrics_payload: dict[str, Any] = dict(payload)
    event_type = metrics_payload.get("event_type") or metrics_payload.get("type")
    if event_type:
        metrics_payload.setdefault("event_type", event_type)
        metrics_payload.setdefault("type", event_type)
    else:
        metrics_payload.setdefault("event_type", "metric")
        metrics_payload.setdefault("type", "metric")

    caught_exc: Exception | None = None
    try:
        event_store.append_metric(bot_id=bot_id, metrics=metrics_payload)
        return
    except TypeError:
        try:
            event_store.append_metric(bot_id, metrics_payload)
            return
        except Exception as exc:  # fall through to shared handler
            caught_exc = exc
    except Exception as exc:
        caught_exc = exc

    if caught_exc is None:
        return

    if logger is not None:
        if hasattr(logger, "log_event"):
            from bot_v2.monitoring.system.logger import LogLevel

            logger.log_event(
                LogLevel.DEBUG,
                "metric_emit_failed",
                "Failed to emit metric to event store",
                error=str(caught_exc),
                bot_id=bot_id,
                metric_type=payload.get("event_type", "unknown"),
            )
        else:  # pragma: no cover - exercised with standard logging.Logger
            try:
                logger.debug(
                    "Failed to emit metric to event store: %s (bot_id=%s, metric_type=%s)",
                    caught_exc,
                    bot_id,
                    payload.get("event_type", "unknown"),
                )
            except Exception:
                pass
    if raise_on_error:
        raise caught_exc


__all__ = ["emit_metric"]
