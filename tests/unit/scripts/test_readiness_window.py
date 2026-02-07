from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import scripts.readiness_window as readiness_window


def _create_events_db(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT
            );
            """
        )
        connection.commit()


def _insert_event(db_path: Path, *, timestamp: str, event_type: str = "api_error") -> None:
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (timestamp, event_type) VALUES (?, ?)",
            (timestamp, event_type),
        )
        connection.commit()


def _freeze_time(monkeypatch, fixed_now: datetime) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(readiness_window, "datetime", FixedDateTime)


def test_parse_timestamp_normalizes_to_utc() -> None:
    parsed_with_offset = readiness_window._parse_timestamp("2026-01-01T19:00:00-05:00")
    assert parsed_with_offset == datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert parsed_with_offset.tzinfo == timezone.utc

    parsed_naive = readiness_window._parse_timestamp("2026-01-02 00:00:00")
    assert parsed_naive == datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert parsed_naive.tzinfo == timezone.utc


def test_main_clears_on_exact_boundary_at_midnight(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixed_now = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    _freeze_time(monkeypatch, fixed_now)

    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    _insert_event(db_path, timestamp="2026-01-01 23:00:00")

    result = readiness_window.main(["--db", str(db_path), "--hours", "1"])

    output = capsys.readouterr().out
    assert result == 0
    assert "window_clears_at_utc: 2026-01-02 00:00:00Z" in output
    assert "time_until_clear: 0s (cleared)" in output


def test_main_pending_just_before_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixed_now = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    _freeze_time(monkeypatch, fixed_now)

    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    _insert_event(db_path, timestamp="2026-01-01T23:00:01Z")

    result = readiness_window.main(["--db", str(db_path), "--hours", "1"])

    output = capsys.readouterr().out
    assert result == 0
    assert "time_until_clear:" in output
    assert "(pending)" in output


def test_main_clears_just_after_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixed_now = datetime(2026, 1, 2, 0, 0, 2, tzinfo=timezone.utc)
    _freeze_time(monkeypatch, fixed_now)

    db_path = tmp_path / "events.db"
    _create_events_db(db_path)
    _insert_event(db_path, timestamp="2026-01-01T23:00:01Z")

    result = readiness_window.main(["--db", str(db_path), "--hours", "1"])

    output = capsys.readouterr().out
    assert result == 0
    assert "time_until_clear:" in output
    assert "(cleared)" in output
