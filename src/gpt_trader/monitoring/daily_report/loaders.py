"""Data loading helpers for daily report generation."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from gpt_trader.utilities.datetime_helpers import normalize_to_utc, to_iso_utc

from .logging_utils import logger  # naming: allow

_TERMINAL_ORDER_STATUSES = {
    "filled",
    "cancelled",
    "canceled",
    "rejected",
    "expired",
    "failed",
}


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return normalize_to_utc(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, OverflowError, ValueError, TypeError):
            return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            return normalize_to_utc(dt)
        except ValueError:
            pass
        try:
            return datetime.fromtimestamp(float(cleaned), tz=UTC)
        except (OSError, OverflowError, ValueError, TypeError):
            return None
    return None


def _normalize_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    fallback_timestamp: Any | None = None,
) -> tuple[dict[str, Any], datetime] | None:
    if not event_type:
        event_type = "unknown"
    if "data" in payload and isinstance(payload["data"], dict) and len(payload) == 1:
        payload = cast(dict[str, Any], payload["data"])
    event = dict(payload)
    event["type"] = event_type
    ts = _parse_timestamp(payload.get("timestamp"))
    if ts is None and fallback_timestamp is not None:
        ts = _parse_timestamp(fallback_timestamp)
    if ts is None:
        return None
    event["timestamp"] = to_iso_utc(ts)
    return event, ts


def _load_events_from_jsonl(events_file: Path, cutoff: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with open(events_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw_event = json.loads(line)
                except Exception as exc:
                    logger.debug(f"Failed to parse event: {exc}")
                    continue
                if not isinstance(raw_event, dict):
                    continue
                event_type = raw_event.get("type") or raw_event.get("event_type")
                normalized = _normalize_event(str(event_type or ""), raw_event)
                if normalized is None:
                    continue
                event, ts = normalized
                if ts >= cutoff:
                    events.append(event)
    except Exception as exc:
        logger.error(f"Failed to load events: {exc}")
    return events


def _load_events_from_db(events_db: Path, cutoff: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(events_db))
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            SELECT timestamp, event_type, payload
            FROM events
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        for row in cursor:
            try:
                payload = json.loads(row["payload"])
            except Exception as exc:
                logger.debug(f"Failed to parse event payload: {exc}")
                continue
            if not isinstance(payload, dict):
                continue
            event_type = str(row["event_type"] or "")
            normalized = _normalize_event(
                event_type,
                payload,
                fallback_timestamp=row["timestamp"],
            )
            if normalized is None:
                continue
            event, ts = normalized
            if ts >= cutoff:
                events.append(event)
    except Exception as exc:
        logger.error(f"Failed to load events from DB: {exc}")
    finally:
        if connection is not None:
            connection.close()
    return events


def _load_latest_metrics_from_db(events_db: Path) -> dict[str, Any]:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(events_db))
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            SELECT payload FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT 250
            """,
            ("metric",),
        )
        fallback_metrics: dict[str, Any] | None = None
        for row in cursor:
            try:
                payload = json.loads(row["payload"])
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue
            if fallback_metrics is None:
                fallback_metrics = metrics
            if metrics.get("event_type") in (None, "cycle_metrics"):
                return metrics
        return fallback_metrics or {}
    except Exception as exc:
        logger.error(f"Failed to load metrics from DB: {exc}")
        return {}
    finally:
        if connection is not None:
            connection.close()


def load_metrics(metrics_file: Path) -> dict[str, Any]:
    """Load aggregated metrics from JSON."""
    if not metrics_file.exists():
        events_db = metrics_file.parent / "events.db"
        if events_db.exists():
            metrics = _load_latest_metrics_from_db(events_db)
            if metrics:
                return metrics
        logger.warning(f"Metrics file not found: {metrics_file}")
        return {}

    try:
        with open(metrics_file) as f:
            return cast(dict[str, Any], json.load(f))
    except Exception as exc:
        logger.error(f"Failed to load metrics: {exc}")
        return {}


def load_events_since(events_file: Path, cutoff: datetime) -> list[dict[str, Any]]:
    """Load JSONL events since the cutoff timestamp."""
    cutoff_utc = normalize_to_utc(cutoff)
    events_db = events_file.with_name("events.db")
    if events_db.exists():
        events = _load_events_from_db(events_db, cutoff_utc)
        logger.info(f"Loaded {len(events)} events since {cutoff_utc}")
        return events

    if events_file.exists():
        events = _load_events_from_jsonl(events_file, cutoff_utc)
        logger.info(f"Loaded {len(events)} events since {cutoff_utc}")
        return events

    logger.warning(f"Events file not found: {events_file}")
    return []


def load_unfilled_orders_count(
    orders_file: Path,
    *,
    as_of: datetime | None = None,
    alert_seconds: int | None = None,
) -> int:
    """Count orders that remain open longer than the alert threshold."""
    if not orders_file.exists():
        return 0
    if alert_seconds is None:
        raw = os.getenv("RISK_UNFILLED_ORDER_ALERT_SECONDS", "").strip()
        try:
            alert_seconds = int(raw) if raw else 300
        except ValueError:
            alert_seconds = 300
    if alert_seconds <= 0:
        return 0
    if as_of is None:
        as_of = datetime.now(UTC)
    cutoff = normalize_to_utc(as_of) - timedelta(seconds=alert_seconds)

    count = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(orders_file))
        connection.row_factory = sqlite3.Row
        cursor = connection.execute("SELECT status, created_at FROM orders")
        for row in cursor:
            status = str(row["status"] or "").lower()
            if status in _TERMINAL_ORDER_STATUSES:
                continue
            created_at = _parse_timestamp(row["created_at"])
            if created_at is None or created_at <= cutoff:
                count += 1
    except Exception as exc:
        logger.error(f"Failed to load orders from DB: {exc}")
    finally:
        if connection is not None:
            connection.close()
    return count


__all__ = ["load_metrics", "load_events_since", "load_unfilled_orders_count"]
