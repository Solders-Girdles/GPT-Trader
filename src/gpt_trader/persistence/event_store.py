"""
Event persistence layer.
"""

from typing import Any


class EventStore:
    def __init__(self, root: Any | None = None):
        self.root = root
        self.events: list[dict[str, Any]] = []

    @property
    def path(self) -> Any | None:
        if self.root:
            return self.root / "events.jsonl"
        return None

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "data": data})

    def append_metric(self, bot_id: str = "unknown", metrics: dict[str, Any] | None = None) -> None:
        self.append("metric", {"bot_id": bot_id, "metrics": metrics or {}})

    def append_position(self, bot_id: str, position: dict[str, Any]) -> None:
        self.append("position", {"bot_id": bot_id, "position": position})
