from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.ops import tail_decision_traces


def _create_events_db(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                bot_id TEXT
            );
            """
        )
        connection.commit()


def test_resolve_decision_id_prefers_decision_id() -> None:
    payload = {"decision_id": "decision-1", "client_order_id": "client-1"}
    assert tail_decision_traces._resolve_decision_id(payload) == "decision-1"


def test_resolve_decision_id_falls_back_to_client_order_id() -> None:
    payload = {"client_order_id": "client-2"}
    assert tail_decision_traces._resolve_decision_id(payload) == "client-2"


def test_read_traces_normalizes_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    payload = json.dumps({"symbol": "BTC-USD"})
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload, timestamp) VALUES (?, ?, ?)",
            ("order_decision_trace", payload, "2026-01-01 00:00:00"),
        )
        connection.commit()

    rows = tail_decision_traces._read_traces(db_path, limit=1)
    assert len(rows) == 1
    assert rows[0].timestamp == "2026-01-01T00:00:00+00:00"
