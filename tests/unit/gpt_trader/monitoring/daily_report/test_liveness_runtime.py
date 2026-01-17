"""Tests for daily report liveness/runtime loaders."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from gpt_trader.monitoring.daily_report.loaders import (
    load_liveness_snapshot,
    load_runtime_fingerprint,
)


def _create_events_db(db_path: Path) -> None:
    connection = sqlite3.connect(str(db_path))
    connection.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()


def _insert_event(
    db_path: Path,
    *,
    timestamp: str,
    event_type: str,
    payload: dict[str, object],
) -> None:
    connection = sqlite3.connect(str(db_path))
    connection.execute(
        "INSERT INTO events (timestamp, event_type, payload) VALUES (?, ?, ?)",
        (timestamp, event_type, json.dumps(payload)),
    )
    connection.commit()
    connection.close()


def test_liveness_green(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    db_path = tmp_path / "liveness_green.db"
    _create_events_db(db_path)
    _insert_event(
        db_path,
        timestamp="2024-01-01 11:59:30",
        event_type="heartbeat",
        payload={"timestamp": "2024-01-01T11:59:30Z"},
    )
    _insert_event(
        db_path,
        timestamp="2024-01-01 11:59:40",
        event_type="price_tick",
        payload={"timestamp": "2024-01-01T11:59:40Z"},
    )

    snapshot = load_liveness_snapshot(db_path, now=now)

    assert snapshot is not None
    assert snapshot["status"] == "GREEN"
    assert snapshot["events"]["heartbeat"]["age_seconds"] == 30
    assert snapshot["events"]["price_tick"]["age_seconds"] == 20


def test_liveness_red(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    db_path = tmp_path / "liveness_red.db"
    _create_events_db(db_path)
    _insert_event(
        db_path,
        timestamp="2024-01-01 11:50:00",
        event_type="heartbeat",
        payload={"timestamp": "2024-01-01T11:50:00Z"},
    )

    snapshot = load_liveness_snapshot(db_path, now=now, max_age_seconds=300)

    assert snapshot is not None
    assert snapshot["status"] == "RED"
    assert snapshot["events"]["heartbeat"]["age_seconds"] == 600


def test_liveness_unknown_when_missing_events(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    db_path = tmp_path / "liveness_unknown.db"
    _create_events_db(db_path)

    snapshot = load_liveness_snapshot(db_path, now=now)

    assert snapshot is not None
    assert snapshot["status"] == "UNKNOWN"


def test_runtime_fingerprint_parses_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_fingerprint.db"
    _create_events_db(db_path)
    _insert_event(
        db_path,
        timestamp="2024-01-01 11:00:00",
        event_type="runtime_start",
        payload={
            "build_sha": "abc123",
            "python_version": "3.12.0",
            "pid": 1234,
            "timestamp": "2024-01-01T11:00:00Z",
        },
    )

    runtime = load_runtime_fingerprint(db_path)

    assert runtime is not None
    assert runtime["event_id"] == 1
    assert runtime["build_sha"] == "abc123"
    assert runtime["timestamp"] == "2024-01-01T11:00:00+00:00"


def test_runtime_fingerprint_does_not_echo_secrets(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_fingerprint_secret.db"
    _create_events_db(db_path)
    _insert_event(
        db_path,
        timestamp="2024-01-01 11:00:00",
        event_type="runtime_start",
        payload={"build_sha": "abc123", "private_key": "SECRET"},
    )

    runtime = load_runtime_fingerprint(db_path)

    assert runtime is not None
    assert "SECRET" not in json.dumps(runtime)
