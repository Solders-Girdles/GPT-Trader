"""Test helpers for recording EventStore interactions."""

from __future__ import annotations

from typing import Any


class RecordingEventStore:
    """In-memory stand-in for EventStore used in unit tests."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def append_metric(self, *args: Any, **kwargs: Any) -> None:
        bot_id = kwargs.get("bot_id")
        metrics = kwargs.get("metrics")

        if args:
            if bot_id is None:
                bot_id = args[0]
            if metrics is None and len(args) > 1:
                metrics = args[1]

        if metrics is None:
            metrics = {
                key: value for key, value in kwargs.items() if key not in {"bot_id", "metrics"}
            }

        self.metrics.append({"bot_id": bot_id, "metrics": metrics})
