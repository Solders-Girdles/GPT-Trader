"""SQLite database engine for event persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from gpt_trader.persistence.durability import (
    WriteError,
    WriteResult,
    check_sqlite_integrity,
    repair_sqlite_database,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="database_engine")


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
        return self._local.connection  # type: ignore[no-any-return]

    def initialize(self, *, check_integrity: bool = True, auto_repair: bool = False) -> None:
        """
        Initialize database schema.

        Args:
            check_integrity: Run integrity check on existing database
            auto_repair: Attempt automatic repair if corruption detected
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Ensure parent directory exists
            self._database_path.parent.mkdir(parents=True, exist_ok=True)

            # Check integrity of existing database
            if check_integrity and self._database_path.exists():
                is_ok, issues = check_sqlite_integrity(self._database_path)
                if not is_ok:
                    logger.error(
                        "Database corruption detected",
                        operation="database_init",
                        path=str(self._database_path),
                        issues=issues[:5],  # Log first 5 issues
                    )
                    if auto_repair:
                        if repair_sqlite_database(self._database_path):
                            logger.info(
                                "Database repaired successfully",
                                operation="database_init",
                            )
                        else:
                            logger.error(
                                "Database repair failed",
                                operation="database_init",
                            )

            connection = self._get_connection()
            schema = self._load_schema()
            connection.executescript(schema)
            self._initialized = True

            logger.debug(
                "Database initialized",
                operation="database_init",
                path=str(self._database_path),
            )

    def _load_schema(self) -> str:
        """Load SQL schema from embedded resource."""
        schema_path = Path(__file__).parent / "schema.sql"
        return schema_path.read_text()

    def write_event(
        self,
        event_type: str,
        data: dict[str, Any],
        bot_id: str | None = None,
        *,
        raise_on_error: bool = False,
    ) -> WriteResult:
        """
        Write event to database.

        Args:
            event_type: Type of event
            data: Event payload
            bot_id: Optional bot identifier for indexing
            raise_on_error: If True, raise WriteError on failure

        Returns:
            WriteResult with success status and row_id

        Raises:
            WriteError: If raise_on_error is True and write fails
        """
        try:
            connection = self._get_connection()
            payload = json.dumps(data)
            cursor = connection.execute(
                """
                INSERT INTO events (event_type, payload, bot_id)
                VALUES (?, ?, ?)
                """,
                (event_type, payload, bot_id),
            )
            row_id = cursor.lastrowid or 0

            logger.debug(
                "Event written",
                operation="write_event",
                event_type=event_type,
                row_id=row_id,
            )

            return WriteResult.ok(row_id=row_id)

        except sqlite3.Error as e:
            error_msg = f"Database write failed: {e}"
            logger.error(
                error_msg,
                operation="write_event",
                event_type=event_type,
                error=str(e),
            )
            if raise_on_error:
                raise WriteError(error_msg) from e
            return WriteResult.fail(error_msg)

        except (TypeError, ValueError) as e:
            error_msg = f"JSON serialization failed: {e}"
            logger.error(
                error_msg,
                operation="write_event",
                event_type=event_type,
                error=str(e),
            )
            if raise_on_error:
                raise WriteError(error_msg) from e
            return WriteResult.fail(error_msg)

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

    def read_recent_events_by_type(self, event_type: str, count: int) -> list[dict[str, Any]]:
        """
        Read most recent events for a specific type.

        Args:
            event_type: Event type filter
            count: Maximum number of events to return

        Returns:
            List of recent events in chronological order
        """
        if count <= 0:
            return []
        connection = self._get_connection()
        cursor = connection.execute(
            """
            SELECT event_type, payload FROM events
            WHERE event_type = ?
            ORDER BY id DESC LIMIT ?
            """,
            (event_type, count),
        )
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
        return int(row[0]) if row else 0

    def prune_by_count(self, max_rows: int) -> int:
        """
        Prune database to keep only the most recent events.

        Deletes the oldest events, keeping only max_rows newest events.
        This is a safe operation that prevents unbounded database growth.

        Args:
            max_rows: Maximum number of rows to keep

        Returns:
            Number of rows deleted
        """
        if max_rows <= 0:
            return 0

        connection = self._get_connection()

        # Get current count
        current_count = self.event_count()
        if current_count <= max_rows:
            return 0

        # Delete oldest events (those with the smallest IDs)
        rows_to_delete = current_count - max_rows
        cursor = connection.execute(
            """
            DELETE FROM events WHERE id IN (
                SELECT id FROM events ORDER BY id ASC LIMIT ?
            )
            """,
            (rows_to_delete,),
        )
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
