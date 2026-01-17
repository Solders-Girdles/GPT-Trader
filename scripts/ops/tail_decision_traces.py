#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionTraceRow:
    timestamp: str
    symbol: str | None
    side: str | None
    reason: str | None
    decision_id: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the most recent order_decision_trace rows from runtime_data/<profile>/events.db"
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of rows (default: 10)")
    return parser.parse_args()


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_decision_id(payload: dict[str, Any]) -> str | None:
    decision_id = _coerce_str(payload.get("decision_id"))
    if decision_id:
        return decision_id
    return _coerce_str(payload.get("client_order_id"))


def _parse_timestamp(value: str) -> str:
    raw = value.strip()
    if not raw:
        return value
    try:
        # events.db stores "YYYY-MM-DD HH:MM:SS"
        dt = datetime.fromisoformat(raw.replace(" ", "T"))
        return dt.isoformat()
    except ValueError:
        return value


def _read_traces(events_db: Path, limit: int) -> list[DecisionTraceRow]:
    if limit <= 0:
        return []

    connection = sqlite3.connect(str(events_db))
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT timestamp, payload
            FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            ("order_decision_trace", limit),
        )
        rows: list[DecisionTraceRow] = []
        for record in cursor:
            timestamp = _parse_timestamp(str(record["timestamp"] or ""))
            payload_raw = record["payload"]
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            rows.append(
                DecisionTraceRow(
                    timestamp=timestamp,
                    symbol=_coerce_str(payload.get("symbol")),
                    side=_coerce_str(payload.get("side")),
                    reason=_coerce_str(payload.get("reason")),
                    decision_id=_resolve_decision_id(payload),
                )
            )
        return rows
    finally:
        connection.close()


def main() -> int:
    args = _parse_args()
    events_db = args.runtime_root / "runtime_data" / args.profile / "events.db"
    if not events_db.exists():
        raise SystemExit(f"events.db not found: {events_db}")

    rows = _read_traces(events_db, args.limit)
    if not rows:
        print("No order_decision_trace rows found.")
        return 1

    print(f"events_db={events_db}")
    print(f"rows={len(rows)}")
    for row in rows:
        symbol = row.symbol or "-"
        side = row.side or "-"
        decision_id = row.decision_id or "-"
        reason = (row.reason or "-").replace("\n", " ").strip()
        print(f"{row.timestamp} | {symbol} | {side} | {decision_id} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
