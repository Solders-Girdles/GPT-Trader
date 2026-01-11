"""Edge-case tests for daily report loaders."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpt_trader.monitoring.daily_report.loaders import load_events_since, load_metrics


def test_load_metrics_missing_file_returns_empty(tmp_path) -> None:
    missing = tmp_path / "metrics.json"

    assert load_metrics(missing) == {}


def test_load_metrics_invalid_json_returns_empty(tmp_path) -> None:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text("{bad json")

    assert load_metrics(metrics_file) == {}


def test_load_events_since_skips_invalid_entries(tmp_path) -> None:
    events_file = tmp_path / "events.jsonl"
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    valid_event = {"timestamp": "2024-01-01T02:00:00Z", "id": "ok"}
    lines = [
        "",
        "{invalid json",
        json.dumps({"id": "missing-ts"}),
        json.dumps({"timestamp": "not-a-time"}),
        json.dumps(valid_event),
    ]
    events_file.write_text("\n".join(lines))

    events = load_events_since(events_file, cutoff)

    assert events == [valid_event]


def test_load_events_since_filters_by_cutoff(tmp_path) -> None:
    events_file = tmp_path / "events.jsonl"
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    before = {"timestamp": "2023-12-31T23:59:59Z", "id": "before"}
    at_cutoff = {"timestamp": "2024-01-01T00:00:00Z", "id": "cutoff"}
    after = {"timestamp": "2024-01-01T01:00:00Z", "id": "after"}
    events_file.write_text(
        "\n".join([json.dumps(before), json.dumps(at_cutoff), json.dumps(after)])
    )

    events = load_events_since(events_file, cutoff)

    assert [event["id"] for event in events] == ["cutoff", "after"]
