#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

try:
    from scripts.ops import formatting
except ModuleNotFoundError:  # pragma: no cover
    # Allow direct script execution (e.g. `python3 scripts/ops/runtime_fingerprint.py ...`).
    import formatting  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print latest runtime_start fingerprint from events.db"
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    parser.add_argument(
        "--min-event-id",
        type=int,
        default=None,
        help="Minimum runtime_start event id required",
    )
    parser.add_argument(
        "--expected-build-sha",
        type=str,
        default=None,
        help="Expected build SHA for runtime_start payload",
    )
    return parser.parse_args()


def _read_latest_runtime_start(events_db: Path) -> dict[str, Any] | None:
    connection = sqlite3.connect(str(events_db))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT id, timestamp, payload
            FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("runtime_start",),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    payload_raw = row["payload"] or "{}"
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    payload.setdefault("timestamp", row["timestamp"])
    payload["event_id"] = row["id"]
    return payload


def _validate_runtime_start(
    payload: dict[str, Any],
    *,
    min_event_id: int | None,
    expected_build_sha: str | None,
) -> tuple[bool, str]:
    event_id = payload.get("event_id")
    if min_event_id is not None:
        if event_id is None or event_id <= min_event_id:
            return False, "runtime_start event_id is not newer than baseline"

    if expected_build_sha is not None:
        payload_sha = payload.get("build_sha")
        if payload_sha != expected_build_sha:
            return False, "runtime_start build_sha does not match expected"

    return True, "ok"


def _print_payload(payload: dict[str, Any]) -> None:
    fields = [
        "event_id",
        "timestamp",
        "profile",
        "bot_id",
        "build_sha",
        "package_version",
        "python_version",
        "pid",
    ]
    for field in fields:
        value = payload.get(field)
        if field == "timestamp":
            value = formatting.format_timestamp(value)
        print(formatting.format_status_line(field, value))


def main() -> int:
    args = _parse_args()
    events_db = args.runtime_root / "runtime_data" / args.profile / "events.db"
    if not events_db.exists():
        print(formatting.format_status_line("error", f"events.db not found: {events_db}"))
        return 1

    try:
        payload = _read_latest_runtime_start(events_db)
    except sqlite3.Error as exc:
        print(formatting.format_status_line("error", f"failed to read events.db: {exc}"))
        return 2

    if payload is None:
        print(formatting.format_status_line("error", "no runtime_start events"))
        return 3

    is_valid, reason = _validate_runtime_start(
        payload,
        min_event_id=args.min_event_id,
        expected_build_sha=args.expected_build_sha,
    )
    if not is_valid:
        print(formatting.format_status_line("error", reason))
        return 4

    _print_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
