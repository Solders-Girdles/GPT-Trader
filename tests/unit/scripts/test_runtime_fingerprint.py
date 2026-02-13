from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from gpt_trader.app.runtime.fingerprint import (
    StartupConfigFingerprint,
    write_startup_config_fingerprint,
)

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
    is_valid, reason, codes = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=2,
        expected_build_sha=None,
    )

    assert is_valid is False
    assert "event_id" in reason
    assert codes == (runtime_fingerprint.REASON_EVENT_ID_NOT_NEWER,)


def test_validate_runtime_start_fails_on_build_sha() -> None:
    payload = {"event_id": 5, "build_sha": "abc"}
    is_valid, reason, codes = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=1,
        expected_build_sha="def",
    )

    assert is_valid is False
    assert "build_sha" in reason
    assert codes == (runtime_fingerprint.REASON_BUILD_SHA_MISMATCH,)


def test_validate_runtime_start_passes_when_matching() -> None:
    payload = {"event_id": 5, "build_sha": "abc"}
    is_valid, reason, codes = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=1,
        expected_build_sha="abc",
    )

    assert is_valid is True
    assert reason == "ok"
    assert codes == ()


def test_validate_runtime_start_returns_multiple_reason_codes() -> None:
    payload = {"event_id": 1, "build_sha": "abc"}
    is_valid, reason, codes = runtime_fingerprint._validate_runtime_start(
        payload,
        min_event_id=5,
        expected_build_sha="def",
    )

    assert is_valid is False
    assert "event_id" in reason
    assert "build_sha" in reason
    assert codes == (
        runtime_fingerprint.REASON_EVENT_ID_NOT_NEWER,
        runtime_fingerprint.REASON_BUILD_SHA_MISMATCH,
    )


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


def test_main_success_does_not_emit_reason_codes(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path
    profile = "canary"
    db_path = runtime_root / "runtime_data" / profile / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_events_db(db_path)

    payload = {
        "profile": profile,
        "build_sha": "abc",
    }
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("runtime_start", json.dumps(payload)),
        )
        connection.commit()

    args = argparse.Namespace(
        profile=profile,
        runtime_root=runtime_root,
        min_event_id=0,
        expected_build_sha="abc",
        config_fingerprint_path=None,
    )
    monkeypatch.setattr(runtime_fingerprint, "_parse_args", lambda: args)
    result = runtime_fingerprint.main()

    output_lines = capsys.readouterr().out.strip().splitlines()
    assert result == 0
    assert not any(line.startswith("reason_codes=") for line in output_lines)


def test_main_failure_emits_reason_codes(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path
    profile = "canary"
    db_path = runtime_root / "runtime_data" / profile / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_events_db(db_path)

    payload = {
        "profile": profile,
        "build_sha": "abc",
    }
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("runtime_start", json.dumps(payload)),
        )
        connection.commit()

    args = argparse.Namespace(
        profile=profile,
        runtime_root=runtime_root,
        min_event_id=100,
        expected_build_sha="def",
        config_fingerprint_path=None,
    )
    monkeypatch.setattr(runtime_fingerprint, "_parse_args", lambda: args)
    result = runtime_fingerprint.main()

    output_lines = [
        line.strip() for line in capsys.readouterr().out.strip().splitlines() if line.strip()
    ]
    assert result == 4
    assert (
        output_lines[-1]
        == "reason_codes=runtime_fingerprint_event_id_not_newer,runtime_fingerprint_build_sha_mismatch"
    )


def test_main_detects_fingerprint_mismatch(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fingerprint mismatch should produce a config reason code."""
    runtime_root = tmp_path
    profile = "canary"
    db_path = runtime_root / "runtime_data" / profile / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_events_db(db_path)

    payload = {
        "profile": profile,
        "build_sha": "abc",
        "config_digest": "def",
    }
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("runtime_start", json.dumps(payload)),
        )
        connection.commit()

    fingerprint_path = runtime_root / "runtime_data" / profile / "startup_config_fingerprint.json"
    write_startup_config_fingerprint(
        fingerprint_path,
        StartupConfigFingerprint(digest="abc", payload={"source": "cli"}),
    )

    args = argparse.Namespace(
        profile=profile,
        runtime_root=runtime_root,
        min_event_id=0,
        expected_build_sha="abc",
        config_fingerprint_path=None,
    )
    monkeypatch.setattr(runtime_fingerprint, "_parse_args", lambda: args)

    result = runtime_fingerprint.main()
    output_lines = [line.strip() for line in capsys.readouterr().out.strip().splitlines() if line.strip()]

    assert result == 4
    assert any("config fingerprint mismatch" in line for line in output_lines)
    assert output_lines[-1].endswith(runtime_fingerprint.REASON_CONFIG_FINGERPRINT_MISMATCH)
