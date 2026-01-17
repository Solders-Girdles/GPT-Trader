from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.ops import runtime_fingerprint


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


def test_read_latest_runtime_start_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)

    payload = {
        "timestamp": 123.0,
        "profile": "dev",
        "bot_id": "dev",
        "build_sha": None,
        "package_version": "0.1.0",
        "python_version": "3.12.0",
        "pid": 999,
    }
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("runtime_start", json.dumps(payload)),
        )
        connection.commit()

    result = runtime_fingerprint._read_latest_runtime_start(db_path)
    assert result is not None
    assert result["build_sha"] is None
    assert result["package_version"] == "0.1.0"
    assert result["event_id"] == 1


def test_read_latest_runtime_start_missing_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)

    result = runtime_fingerprint._read_latest_runtime_start(db_path)
    assert result is None
