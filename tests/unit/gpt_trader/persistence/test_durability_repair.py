"""Regression tests for SQLite database repair."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import gpt_trader.persistence.durability as durability


def _sidecar_paths(database_path: Path) -> tuple[Path, Path]:
    return (
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
    )


def _create_repair_fixture_database(database_path: Path) -> None:
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute("CREATE TABLE repair_items (id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("INSERT INTO repair_items VALUES (1, 'kept')")


def test_repair_sqlite_database_aborts_when_backup_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "events.db"
    _create_repair_fixture_database(database_path)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(durability.shutil, "copy2", fail_copy)

    assert durability.repair_sqlite_database(database_path) is False
    assert database_path.exists()
    assert not database_path.with_suffix(".corrupted").exists()
    with sqlite3.connect(str(database_path)) as connection:
        assert connection.execute("SELECT name FROM repair_items").fetchone() == ("kept",)


def test_repair_sqlite_database_removes_stale_wal_sidecars(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    _create_repair_fixture_database(database_path)
    for sidecar_path in _sidecar_paths(database_path):
        sidecar_path.write_bytes(b"stale")

    assert durability.repair_sqlite_database(database_path) is True

    assert all(not sidecar_path.exists() for sidecar_path in _sidecar_paths(database_path))
    with sqlite3.connect(str(database_path)) as connection:
        assert connection.execute("SELECT name FROM repair_items").fetchone() == ("kept",)


def test_repair_sqlite_database_preserves_uncheckpointed_wal_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE repair_items (id INTEGER PRIMARY KEY, name TEXT)")
        connection.commit()
        connection.execute("INSERT INTO repair_items VALUES (1, 'from-wal')")
        connection.commit()
        assert _sidecar_paths(database_path)[0].stat().st_size > 0
    finally:
        connection.close()

    assert durability.repair_sqlite_database(database_path) is True

    with sqlite3.connect(str(database_path)) as repaired:
        assert repaired.execute("SELECT name FROM repair_items").fetchone() == ("from-wal",)


def test_repair_sqlite_database_copies_common_columns_from_stale_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.db"
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("order_submitted", '{"symbol": "BTC-USD"}'),
        )

    assert durability.repair_sqlite_database(database_path) is True

    with sqlite3.connect(str(database_path)) as repaired:
        row = repaired.execute(
            "SELECT id, event_type, payload, timestamp, bot_id FROM events"
        ).fetchone()

    assert row[:3] == (1, "order_submitted", '{"symbol": "BTC-USD"}')
    assert row[3] is not None
    assert row[4] is None
