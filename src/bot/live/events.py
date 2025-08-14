from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    SELECTION = "selection"
    REBALANCE = "rebalance"
    RISK = "risk"
    PERFORMANCE = "performance"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]


class EventBus:
    """Simple in-process event bus for publishing system events.

    Subscribers register callbacks by event type. Callbacks should be fast and non-blocking.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event: Event) -> None:
        for cb in self._subscribers.get(event.type, []):
            try:
                cb(event)
            except Exception:
                # swallow to avoid cascading failures
                pass
