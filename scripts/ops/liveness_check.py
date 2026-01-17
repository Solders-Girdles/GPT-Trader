#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable

DEFAULT_EVENT_TYPES = ("heartbeat", "price_tick")
DEFAULT_MAX_AGE_SECONDS = 300
STALE_AGE_SECONDS = 999999


@dataclass(frozen=True)
class EventAge:
    event_type: str
    last_ts: str
    age_seconds: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check canary liveness from heartbeat/price_tick events."
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=DEFAULT_MAX_AGE_SECONDS,
        help="Max age in seconds before marking RED (default: 300)",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        default=None,
        help="Event type to consider (repeatable)",
    )
    return parser.parse_args()


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace(" ", "T"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.isoformat()


def _fetch_latest_timestamp(connection: sqlite3.Connection, event_type: str) -> str | None:
    cursor = connection.execute(
        "SELECT max(timestamp) FROM events WHERE event_type = ?", (event_type,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def check_liveness(
    events_db: Path,
    event_types: Iterable[str],
    max_age_seconds: int,
    *,
    now: datetime | None = None,
) -> tuple[list[EventAge], bool]:
    if not events_db.exists():
        raise FileNotFoundError(f"events.db not found: {events_db}")

    connection = sqlite3.connect(str(events_db))
    try:
        rows: list[EventAge] = []
        now_ts = now or datetime.now(timezone.utc)
        for event_type in event_types:
            raw_ts = _fetch_latest_timestamp(connection, event_type)
            parsed = _parse_timestamp(raw_ts)
            if parsed is None:
                rows.append(
                    EventAge(
                        event_type=event_type,
                        last_ts="-",
                        age_seconds=STALE_AGE_SECONDS,
                    )
                )
                continue
            age_seconds = int((now_ts - parsed).total_seconds())
            rows.append(
                EventAge(
                    event_type=event_type,
                    last_ts=_format_timestamp(parsed),
                    age_seconds=age_seconds,
                )
            )
        is_green = any(row.age_seconds <= max_age_seconds for row in rows)
        return rows, is_green
    finally:
        connection.close()


def main() -> int:
    args = _parse_args()
    event_types = args.event_type or list(DEFAULT_EVENT_TYPES)

    events_db = args.runtime_root / "runtime_data" / args.profile / "events.db"
    print(f"events_db={events_db}")

    try:
        rows, is_green = check_liveness(
            events_db,
            event_types,
            args.max_age_seconds,
        )
    except (sqlite3.Error, FileNotFoundError) as exc:
        print("liveness_status=ERROR")
        print(f"error={exc}")
        return 2

    for row in rows:
        print(f"{row.event_type} last_ts={row.last_ts} age_seconds={row.age_seconds}")

    status = "GREEN" if is_green else "RED"
    print(f"liveness_status={status}")
    return 0 if is_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
