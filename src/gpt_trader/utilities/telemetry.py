from typing import Any
import logging


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
    if event_store is not None and hasattr(event_store, "append_metric"):
        event_store.append_metric(bot_id=bot_id, metrics=metrics)
