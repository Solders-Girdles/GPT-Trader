#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


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
        if value is None or value == "":
            value = "-"
        print(f"{field}={value}")


def main() -> int:
    args = _parse_args()
    events_db = args.runtime_root / "runtime_data" / args.profile / "events.db"
    if not events_db.exists():
        print(f"error=events.db not found: {events_db}")
        return 1

    try:
        payload = _read_latest_runtime_start(events_db)
    except sqlite3.Error as exc:
        print(f"error=failed to read events.db: {exc}")
        return 2

    if payload is None:
        print("error=no runtime_start events")
        return 3

    _print_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
