"""SQLite database engine for event persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class DatabaseEngine:
    """
    SQLite database engine for event storage.

    Handles:
    - Connection management with thread safety
    - Schema initialization
    - Event read/write operations

    Uses WAL mode for concurrent read performance and thread-local
    connections for thread safety.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize database engine.

        Args:
            database_path: Path to SQLite database file
        """
        self._database_path = database_path
        self._lock = threading.Lock()
        self._initialized = False
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            connection = sqlite3.connect(
                str(self._database_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.row_factory = sqlite3.Row
            self._local.connection = connection
        return self._local.connection

    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Ensure parent directory exists
            self._database_path.parent.mkdir(parents=True, exist_ok=True)

            connection = self._get_connection()
            schema = self._load_schema()
            connection.executescript(schema)
            self._initialized = True

    def _load_schema(self) -> str:
        """Load SQL schema from embedded resource."""
        schema_path = Path(__file__).parent / "schema.sql"
        return schema_path.read_text()

    def write_event(
        self,
        event_type: str,
        data: dict[str, Any],
        bot_id: str | None = None,
    ) -> int:
        """
        Write event to database.

        Args:
            event_type: Type of event
            data: Event payload
            bot_id: Optional bot identifier for indexing

        Returns:
            Row ID of inserted event
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            INSERT INTO events (event_type, payload, bot_id)
            VALUES (?, ?, ?)
            """,
            (event_type, json.dumps(data), bot_id),
        )
        return cursor.lastrowid or 0

    def read_all_events(self) -> list[dict[str, Any]]:
        """
        Read all events from database in insertion order.

        Returns:
            List of events as {"type": str, "data": dict}
        """
        connection = self._get_connection()
        cursor = connection.execute("SELECT event_type, payload FROM events ORDER BY id ASC")
        return [{"type": row["event_type"], "data": json.loads(row["payload"])} for row in cursor]

    def read_recent_events(self, count: int) -> list[dict[str, Any]]:
        """
        Read most recent events.

        Args:
            count: Maximum number of events to return

        Returns:
            List of recent events in chronological order
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            SELECT event_type, payload FROM events
            ORDER BY id DESC LIMIT ?
            """,
            (count,),
        )
        # Reverse to get chronological order
        rows = list(cursor)
        return [
            {"type": row["event_type"], "data": json.loads(row["payload"])}
            for row in reversed(rows)
        ]

    def read_events_by_bot(self, bot_id: str) -> list[dict[str, Any]]:
        """
        Read events for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            List of events for the bot
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            SELECT event_type, payload FROM events
            WHERE bot_id = ? ORDER BY id ASC
            """,
            (bot_id,),
        )
        return [{"type": row["event_type"], "data": json.loads(row["payload"])} for row in cursor]

    def event_count(self) -> int:
        """
        Get total number of events in database.

        Returns:
            Total event count
        """
        connection = self._get_connection()
        cursor = connection.execute("SELECT COUNT(*) FROM events")
        row = cursor.fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
