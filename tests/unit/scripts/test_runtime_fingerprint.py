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


def test_validate_runtime_start_fails_on_event_id() -> None:
    payload = {"event_id": 1, "build_sha": "abc"}
    is_valid, reason = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=2,
        expected_build_sha=None,
    )

    assert is_valid is False
    assert "event_id" in reason


def test_validate_runtime_start_fails_on_build_sha() -> None:
    payload = {"event_id": 5, "build_sha": "abc"}
    is_valid, reason = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=1,
        expected_build_sha="def",
    )

    assert is_valid is False
    assert "build_sha" in reason


def test_validate_runtime_start_passes_when_matching() -> None:
    payload = {"event_id": 5, "build_sha": "abc"}
    is_valid, reason = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=1,
        expected_build_sha="abc",
    )

    assert is_valid is True
    assert reason == "ok"


def test_print_payload_normalizes_timestamp(capsys) -> None:
    payload = {
        "event_id": 1,
        "timestamp": "2026-02-01 12:00:00",
        "profile": "dev",
    }

    runtime_fingerprint._print_payload(payload)
    output = capsys.readouterr().out.strip().splitlines()
    values = dict(line.split("=", 1) for line in output)

    assert values["timestamp"] == "2026-02-01T12:00:00+00:00"
    assert values["event_id"] == "1"
