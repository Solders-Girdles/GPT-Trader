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

    def store_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Alias for append - stores an event with given type and data."""
        self.append(event_type, data)

    def append_error(
        self,
        error: str | None = None,
        details: dict[str, Any] | None = None,
        *,
        bot_id: str | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Append an error event."""
        error_message = message or error or "unknown_error"
        error_details = context or details or {}
        self.append(
            "error",
            {
                "bot_id": bot_id,
                "error": error_message,
                "details": error_details,
            },
        )

    def append_trade(
        self,
        bot_id_or_trade: str | dict[str, Any],
        trade: dict[str, Any] | None = None,
    ) -> None:
        """Append a trade event."""
        if isinstance(bot_id_or_trade, dict):
            # Called as append_trade(trade_dict)
            self.append("trade", bot_id_or_trade)
        else:
            # Called as append_trade(bot_id, trade_dict)
            self.append("trade", {"bot_id": bot_id_or_trade, **(trade or {})})

    def store(self, event: Any) -> None:
        """Store an event (protocol compliance).

        Satisfies EventStoreProtocol.store() interface.
        """
        if isinstance(event, dict):
            event_type = event.get("type", "unknown")
            data = event.get("data", event)
            self.append(event_type, data)
        else:
            self.append("event", {"payload": event})

    def get_recent(self, count: int = 100) -> list[Any]:
        """Get recent events (protocol compliance).

        Satisfies EventStoreProtocol.get_recent() interface.
        """
        return self.events[-count:] if count > 0 else []
