from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credentials",
    "key_name",
    "passphrase",
    "password",
    "private_key",
    "privatekey",
    "privatekeypem",
    "secret",
    "token",
}

PEM_MARKERS = (
    "BEGIN EC PRIVATE KEY",
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
)


def _is_warn_only() -> bool:
    return os.getenv("GPT_TRADER_PREFLIGHT_WARN_ONLY", "0") == "1"


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_event_store_path(profile: str) -> Path:
    override_root = os.getenv("EVENT_STORE_ROOT")
    if override_root:
        override_path = Path(override_root).expanduser()
        if "runtime_data" not in set(override_path.parts):
            override_path = override_path / "runtime_data" / profile
        return override_path / "events.db"

    runtime_root = Path(os.getenv("GPT_TRADER_RUNTIME_ROOT", ".")).expanduser()
    return runtime_root / "runtime_data" / profile / "events.db"


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _is_redacted(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "[REDACTED]"}
    return False


def _walk_payload(
    payload: Any,
    *,
    path: tuple[str, ...] = (),
) -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            key_path = path + (key_str,)
            key_lower = key_str.lower()
            if key_lower in SENSITIVE_KEYS and not _is_redacted(value):
                findings.append(".".join(key_path))
            findings.extend(_walk_payload(value, path=key_path))
        return findings

    if isinstance(payload, list):
        for index, value in enumerate(payload):
            index_path = path + (f"[{index}]",)
            findings.extend(_walk_payload(value, path=index_path))
        return findings

    if isinstance(payload, str):
        upper_value = payload.upper()
        if any(marker in upper_value for marker in PEM_MARKERS):
            findings.append(".".join(path or ("<payload>",)))
        if "BEARER " in upper_value and "[REDACTED]" not in upper_value:
            findings.append(".".join(path or ("<payload>",)))
    return findings


def _get_latest_runtime_start_id(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT id FROM events WHERE event_type = ? ORDER BY id DESC LIMIT 1",
        ("runtime_start",),
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    return 0


def check_event_store_redaction(checker: PreflightCheck) -> bool:
    """Scan events.db payloads for secret leakage and unredacted credentials."""
    checker.section_header("13. EVENT STORE REDACTION (SECURITY)")

    warn_only = _is_warn_only()
    events_path = _resolve_event_store_path(checker.profile)
    if not events_path.exists():
        message = f"Events DB not found: {events_path}"
        if checker.profile == "dev" or warn_only:
            checker.log_warning(message)
            return True
        checker.log_error(message)
        return False

    min_event_override = os.getenv("GPT_TRADER_EVENT_STORE_REDACTION_MIN_EVENT_ID")
    scan_all = os.getenv("GPT_TRADER_EVENT_STORE_REDACTION_SCAN_ALL") == "1"
    max_rows = _get_env_int("GPT_TRADER_EVENT_STORE_REDACTION_MAX_ROWS", 20000)

    try:
        uri = f"file:{events_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5.0) as connection:
            connection.row_factory = sqlite3.Row
            base_min_id = _get_latest_runtime_start_id(connection)
            if scan_all:
                base_min_id = 0
            min_event_id = base_min_id
            if min_event_override is not None and min_event_override != "":
                try:
                    min_event_id = int(min_event_override)
                except ValueError:
                    min_event_id = base_min_id

            rows = connection.execute(
                """
                SELECT id, event_type, payload
                FROM events
                WHERE id > ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (min_event_id, max_rows),
            ).fetchall()
    except sqlite3.Error as exc:
        message = f"Failed to read events DB: {exc}"
        if warn_only:
            checker.log_warning(message)
            return True
        checker.log_error(message)
        return False

    if not rows:
        checker.log_success("Event store redaction check passed (0 events scanned)")
        return True

    findings: list[tuple[int, str, str]] = []
    for row in rows:
        event_id = int(row["id"])
        event_type = str(row["event_type"])
        payload = _coerce_json(row["payload"])
        paths = sorted(set(_walk_payload(payload)))
        for path in paths:
            findings.append((event_id, event_type, path))

    if not findings:
        checker.log_success(f"Event store redaction check passed ({len(rows)} events scanned)")
        return True

    summary = f"Event store redaction check found {len(findings)} issues"
    if warn_only:
        checker.log_warning(summary)
    else:
        checker.log_error(summary)

    for event_id, event_type, path in findings[:20]:
        message = f"Unredacted secret in event {event_id} ({event_type}) at {path}"
        if warn_only:
            checker.log_warning(message)
        else:
            checker.log_error(message)

    if len(findings) > 20:
        extra = len(findings) - 20
        message = f"{extra} additional redaction findings not shown"
        if warn_only:
            checker.log_warning(message)
        else:
            checker.log_error(message)

    return warn_only


__all__ = ["check_event_store_redaction"]
