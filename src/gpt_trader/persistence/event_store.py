"""
Event persistence layer.
"""

from typing import Any, Dict, Optional


class EventStore:
    def __init__(self, root: Optional[Any] = None):
        self.root = root
        self.events = []

    @property
    def path(self) -> Optional[Any]:
        if self.root:
            return self.root / "events.jsonl"
        return None

    def append(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({"type": event_type, "data": data})

    def append_metric(self, bot_id: str = "unknown", metrics: Dict[str, Any] = None) -> None:
        self.append("metric", {"bot_id": bot_id, "metrics": metrics or {}})

    def append_position(self, bot_id: str, position: Dict[str, Any]) -> None:
        self.append("position", {"bot_id": bot_id, "position": position})
