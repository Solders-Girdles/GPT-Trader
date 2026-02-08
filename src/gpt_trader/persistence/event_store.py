"""
Event persistence layer with optional SQLite durability.
"""

from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Any

from gpt_trader.persistence.database import DatabaseEngine

DEFAULT_CACHE_SIZE = 10_000


class EventStore:
    """
    Event store with optional SQLite persistence.

    Modes:
    - In-memory (root=None): Pure deque storage, identical to legacy behavior
    - Persistent (root provided): Write-through to SQLite with bounded cache

    The `events` property always returns a list for backward compatibility.
    """

    def __init__(
        self,
        root: Path | None = None,
        max_cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """
        Initialize event store.

        Args:
            root: Storage root directory. If None, operates in memory-only mode.
            max_cache_size: Maximum events to keep in memory cache (default: 10,000)
        """
        self._root = root
        self._max_cache_size = max_cache_size
        self._events: deque[dict[str, Any]] = deque(maxlen=max_cache_size)
        self._database: DatabaseEngine | None = None

        if root is not None:
            database_path = root / "events.db"
            self._database = DatabaseEngine(database_path)
            self._database.initialize()
            # Load recent events into cache for restart recovery
            for event in self._database.read_recent_events(max_cache_size):
                self._events.append(event)

    @property
    def root(self) -> Path | None:
        """Storage root directory."""
        return self._root

    @property
    def path(self) -> Path | None:
        """Legacy path property for backward compatibility."""
        warnings.warn(
            "EventStore.path is deprecated; use EventStore.root and store SQLite data in events.db.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root:
            return self._root / "events.jsonl"
        return None

    @property
    def events(self) -> list[dict[str, Any]]:
        """
        Direct access to events list.

        Returns a list copy of the internal deque for backward compatibility
        with tests that use store.events[0], len(store.events), etc.
        """
        warnings.warn(
            "EventStore.events is deprecated; use EventStore.list_events().",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self._events)

    def list_events(self, count: int | None = None) -> list[dict[str, Any]]:
        """Return a snapshot of events.

        In persistent mode, reads from the database to include all events,
        not just the in-memory cache.

        Args:
            count: If provided, return only the most recent count events.
        """
        if count is not None:
            return self.get_recent(count)
        if self._database is not None:
            return self._database.read_all_events()
        return list(self._events)

    def list_events_by_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """Return events filtered by trading symbol.

        In persistent mode, uses an indexed database query to avoid
        reading and deserializing all events.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD").
        """
        if self._database is not None:
            return self._database.read_events_by_symbol(symbol)
        return [event for event in self._events if event.get("data", {}).get("symbol") == symbol]

    def _extract_bot_id(self, data: dict[str, Any]) -> str | None:
        """Extract bot_id from event data for database indexing."""
        # Direct field
        if "bot_id" in data:
            return str(data["bot_id"]) if data["bot_id"] is not None else None
        # Nested in position/trade payloads
        for key in ("position", "trade"):
            if key in data and isinstance(data[key], dict):
                if "bot_id" in data[key]:
                    return str(data[key]["bot_id"]) if data[key]["bot_id"] is not None else None
        return None

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Append an event to the store.

        In persistent mode, writes through to SQLite before adding to cache.
        """
        event = {"type": event_type, "data": data}

        # Write to database first (if persistent mode)
        if self._database is not None:
            bot_id = self._extract_bot_id(data)
            self._database.write_event(event_type, data, bot_id)

        # Always update in-memory cache (deque auto-evicts oldest if full)
        self._events.append(event)

    def append_metric(self, bot_id: str = "unknown", metrics: dict[str, Any] | None = None) -> None:
        """Append a metric event."""
        self.append("metric", {"bot_id": bot_id, "metrics": metrics or {}})

    def append_position(self, bot_id: str, position: dict[str, Any]) -> None:
        """Append a position event."""
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
        if count <= 0:
            return []
        if self._database is not None:
            return self._database.read_recent_events(count)
        events_list = list(self._events)
        return events_list[-count:]

    def get_recent_by_type(self, event_type: str, count: int = 100) -> list[Any]:
        """Get recent events filtered by type.

        Uses the database when available to avoid missing events outside the cache.
        """
        if count <= 0:
            return []
        if self._database is not None:
            return self._database.read_recent_events_by_type(event_type, count)
        filtered = [event for event in self._events if event.get("type") == event_type]
        return filtered[-count:]

    def get_cache_size(self) -> int:
        """Get current number of events in the in-memory cache."""
        return len(self._events)

    def get_cache_max_size(self) -> int:
        """Get maximum size of the in-memory cache."""
        return self._max_cache_size

    def get_cache_fill_ratio(self) -> float:
        """Get cache fill ratio (0.0 to 1.0).

        Returns:
            Ratio of current cache size to max cache size.
        """
        if self._max_cache_size == 0:
            return 0.0
        return len(self._events) / self._max_cache_size

    def prune(self, max_rows: int = 1_000_000) -> int:
        """
        Prune the event store to prevent unbounded growth.

        In persistent mode, deletes oldest events from SQLite keeping max_rows.
        In memory-only mode, the deque already limits cache size, so this is a no-op.

        Args:
            max_rows: Maximum number of events to keep in database (default: 1M)

        Returns:
            Number of events pruned (0 in memory-only mode)
        """
        if self._database is None:
            return 0
        return self._database.prune_by_count(max_rows)

    def close(self) -> None:
        """Close database connection (if persistent mode)."""
        if self._database is not None:
            self._database.close()

    def __enter__(self) -> EventStore:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - ensures database connection is closed."""
        self.close()
