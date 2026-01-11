from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.persistence.database import DatabaseEngine
from gpt_trader.persistence.durability import WriteError


def test_write_event_json_failure_returns_fail(tmp_path: Path) -> None:
    engine = DatabaseEngine(tmp_path / "events.db")
    try:
        result = engine.write_event("event", {"value": Decimal("1.0")})
        assert result.success is False
        assert result.error and "JSON serialization failed" in result.error
    finally:
        engine.close()


def test_write_event_json_failure_raises(tmp_path: Path) -> None:
    engine = DatabaseEngine(tmp_path / "events.db")
    try:
        with pytest.raises(WriteError):
            engine.write_event(
                "event",
                {"value": Decimal("1.0")},
                raise_on_error=True,
            )
    finally:
        engine.close()


def test_write_event_sqlite_error_returns_fail(tmp_path: Path, monkeypatch) -> None:
    engine = DatabaseEngine(tmp_path / "events.db")

    class StubConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.Error("boom")

    monkeypatch.setattr(engine, "_get_connection", lambda: StubConnection())

    result = engine.write_event("event", {"value": "ok"})
    assert result.success is False
    assert result.error and "Database write failed" in result.error


def test_write_event_sqlite_error_raises(tmp_path: Path, monkeypatch) -> None:
    engine = DatabaseEngine(tmp_path / "events.db")

    class StubConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.Error("boom")

    monkeypatch.setattr(engine, "_get_connection", lambda: StubConnection())

    with pytest.raises(WriteError):
        engine.write_event("event", {"value": "ok"}, raise_on_error=True)
