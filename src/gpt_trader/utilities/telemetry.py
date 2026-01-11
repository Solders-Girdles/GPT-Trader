from typing import Any

from gpt_trader.utilities.datetime_helpers import utc_now_iso


def emit_metric(
    event_store: Any,
    bot_id: str,
    metrics: dict[str, Any],
    *,
    logger: Any | None = None,
) -> None:
    """
    Emit a metric event to the event store.

    Args:
        event_store: EventStore instance to persist the metric
        bot_id: Bot identifier
        metrics: Dictionary of metric data
        logger: Optional logger for additional logging
    """
    if event_store is None or not hasattr(event_store, "append_metric"):
        return

    payload = dict(metrics)
    if "event_type" not in payload:
        if "type" in payload:
            payload["event_type"] = payload.pop("type")
        else:
            payload["event_type"] = "metric"

    payload.setdefault("timestamp", utc_now_iso())

    try:
        event_store.append_metric(bot_id=bot_id, metrics=payload)
    except Exception as exc:
        if logger is not None:
            logger.debug(
                "Failed to emit metric",
                event_type=payload.get("event_type"),
                bot_id=bot_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
