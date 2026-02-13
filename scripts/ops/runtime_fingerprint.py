#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.ops import formatting
except ModuleNotFoundError:  # pragma: no cover
    # Allow direct script execution (e.g. `python3 scripts/ops/runtime_fingerprint.py ...`).
    import formatting  # type: ignore

try:
    from gpt_trader.app.runtime.fingerprint import (
        StartupConfigFingerprint,
        compare_startup_config_fingerprints,
        load_startup_config_fingerprint,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Allow direct script execution from a checkout without `pip install -e .`.
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from gpt_trader.app.runtime.fingerprint import (
        StartupConfigFingerprint,
        compare_startup_config_fingerprints,
        load_startup_config_fingerprint,
    )

REASON_DB_NOT_FOUND = "runtime_fingerprint_events_db_missing"
REASON_DB_READ_ERROR = "runtime_fingerprint_events_db_read_error"
REASON_RUNTIME_START_MISSING = "runtime_fingerprint_runtime_start_missing"
REASON_EVENT_ID_NOT_NEWER = "runtime_fingerprint_event_id_not_newer"
REASON_BUILD_SHA_MISMATCH = "runtime_fingerprint_build_sha_mismatch"
REASON_CONFIG_FINGERPRINT_MISMATCH = "runtime_fingerprint_config_mismatch"


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
        "--config-fingerprint-path",
        type=Path,
        default=None,
        help="Path to persisted startup config fingerprint (default: runtime_data/<profile>/startup_config_fingerprint.json)",
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
    fingerprint_comparison: tuple[bool, str] | None = None,
) -> tuple[bool, str, tuple[str, ...]]:
    event_id = payload.get("event_id")
    reasons: list[str] = []
    reason_codes: list[str] = []

    if min_event_id is not None:
        if event_id is None or event_id <= min_event_id:
            reasons.append("runtime_start event_id is not newer than baseline")
            reason_codes.append(REASON_EVENT_ID_NOT_NEWER)

    if expected_build_sha is not None:
        payload_sha = payload.get("build_sha")
        if payload_sha != expected_build_sha:
            reasons.append("runtime_start build_sha does not match expected")
            reason_codes.append(REASON_BUILD_SHA_MISMATCH)

    if fingerprint_comparison is not None and not fingerprint_comparison[0]:
        reasons.append(fingerprint_comparison[1])
        reason_codes.append(REASON_CONFIG_FINGERPRINT_MISMATCH)

    if not reasons:
        return True, "ok", tuple()
    return False, "; ".join(reasons), tuple(reason_codes)


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
        "config_digest",
    ]
    for field in fields:
        value = payload.get(field)
        if field == "timestamp":
            value = formatting.format_timestamp(value)
        print(formatting.format_status_line(field, value))


def _print_error(message: str, *, reason_codes: tuple[str, ...] = ()) -> None:
    print(formatting.format_status_line("error", message))
    if reason_codes:
        print(formatting.format_reason_codes(reason_codes))


def main() -> int:
    args = _parse_args()
    events_db = args.runtime_root / "runtime_data" / args.profile / "events.db"
    if not events_db.exists():
        _print_error(f"events.db not found: {events_db}", reason_codes=(REASON_DB_NOT_FOUND,))
        return 1

    try:
        payload = _read_latest_runtime_start(events_db)
    except sqlite3.Error as exc:
        _print_error(
            f"failed to read events.db: {exc}",
            reason_codes=(REASON_DB_READ_ERROR,),
        )
        return 2

    if payload is None:
        _print_error("no runtime_start events", reason_codes=(REASON_RUNTIME_START_MISSING,))
        return 3

    fingerprint_path = (
        args.config_fingerprint_path
        or args.runtime_root / "runtime_data" / args.profile / "startup_config_fingerprint.json"
    )
    expected_fingerprint = load_startup_config_fingerprint(fingerprint_path)
    fingerprint_comparison = None
    if expected_fingerprint is not None:
        actual_digest = payload.get("config_digest")
        actual_fingerprint = (
            StartupConfigFingerprint(digest=actual_digest, payload={})
            if actual_digest is not None
            else None
        )
        fingerprint_comparison = compare_startup_config_fingerprints(
            expected_fingerprint, actual_fingerprint
        )

    is_valid, reason, reason_codes = _validate_runtime_start(
        payload,
        min_event_id=args.min_event_id,
        expected_build_sha=args.expected_build_sha,
        fingerprint_comparison=fingerprint_comparison,
    )
    if not is_valid:
        _print_error(reason, reason_codes=reason_codes)
        return 4

    _print_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
