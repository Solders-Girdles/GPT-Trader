"""Data loading helpers for daily report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_utils import logger  # naming: allow


def load_metrics(metrics_file: Path) -> dict[str, Any]:
    """Load aggregated metrics from JSON."""
    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return {}

    try:
        with open(metrics_file) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.error(f"Failed to load metrics: {exc}")
        return {}


def load_events_since(events_file: Path, cutoff: datetime) -> list[dict[str, Any]]:
    """Load JSONL events since the cutoff timestamp."""
    if not events_file.exists():
        logger.warning(f"Events file not found: {events_file}")
        return []

    events: list[dict[str, Any]] = []
    try:
        with open(events_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except Exception as exc:
                    logger.debug(f"Failed to parse event: {exc}")
                    continue
                timestamp = event.get("timestamp")
                if not timestamp:
                    continue
                try:
                    ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                except Exception:
                    continue
                if ts >= cutoff:
                    events.append(event)
    except Exception as exc:
        logger.error(f"Failed to load events: {exc}")

    logger.info(f"Loaded {len(events)} events since {cutoff}")
    return events


__all__ = ["load_metrics", "load_events_since"]
