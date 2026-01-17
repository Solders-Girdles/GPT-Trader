from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import liveness_check


def _write_event(db_path: Path, event_type: str, timestamp: datetime) -> None:
    payload = "{}"
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload, bot_id, timestamp) VALUES (?, ?, ?, ?)",
            (
                event_type,
                payload,
                None,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


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


def test_only_decision_trace_is_red(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    now = datetime.now(timezone.utc)
    _write_event(db_path, "order_decision_trace", now)

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat", "price_tick"],
        max_age_seconds=300,
        now=now,
    )

    assert is_green is False
    assert {row.event_type for row in rows} == {"heartbeat", "price_tick"}
    assert all(row.age_seconds == liveness_check.STALE_AGE_SECONDS for row in rows)


def test_fresh_heartbeat_is_green(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    now = datetime.now(timezone.utc)
    _write_event(db_path, "heartbeat", now - timedelta(seconds=42))

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat", "price_tick"],
        max_age_seconds=300,
        now=now,
    )

    assert is_green is True
    heartbeat = next(row for row in rows if row.event_type == "heartbeat")
    assert heartbeat.age_seconds == 42


def test_fresh_price_tick_is_green(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    now = datetime.now(timezone.utc)
    _write_event(db_path, "price_tick", now - timedelta(seconds=10))

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat", "price_tick"],
        max_age_seconds=300,
        now=now,
    )

    assert is_green is True
    price_tick = next(row for row in rows if row.event_type == "price_tick")
    assert price_tick.age_seconds == 10


def test_both_stale_is_red(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    now = datetime.now(timezone.utc)
    _write_event(db_path, "heartbeat", now - timedelta(seconds=600))
    _write_event(db_path, "price_tick", now - timedelta(seconds=900))

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat", "price_tick"],
        max_age_seconds=300,
        now=now,
    )

    assert is_green is False
    assert {row.event_type for row in rows} == {"heartbeat", "price_tick"}


def test_liveness_min_event_id_requires_newer_event(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    now = datetime.now(timezone.utc)
    _write_event(db_path, "heartbeat", now)

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat"],
        max_age_seconds=300,
        min_event_id=1,
        now=now,
    )

    assert rows[0].event_id == 1
    assert is_green is False

    _write_event(db_path, "heartbeat", now)

    rows, is_green = liveness_check.check_liveness(
        db_path,
        ["heartbeat"],
        max_age_seconds=300,
        min_event_id=1,
        now=now,
    )

    assert rows[0].event_id == 2
    assert is_green is True
