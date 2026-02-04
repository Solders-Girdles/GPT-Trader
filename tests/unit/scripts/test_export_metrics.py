from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts.monitoring import export_metrics


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


def _insert_event(
    db_path: Path,
    event_type: str,
    payload: dict[str, object],
    timestamp: str,
) -> None:
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload, timestamp) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), timestamp),
        )
        connection.commit()


def test_load_metrics_reads_file(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"equity": 123.0, "timestamp": "2025-01-01T00:00:00Z"}))

    result = export_metrics.load_metrics(metrics_path)

    assert result["equity"] == 123.0
    assert result["timestamp"] == "2025-01-01T00:00:00Z"


def test_load_metrics_falls_back_to_db_when_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    events_db = tmp_path / "events.db"
    _create_events_db(events_db)

    _insert_event(
        events_db,
        "metric",
        {"metrics": {"event_type": "cycle_metrics", "equity": 999.0}},
        "2025-01-01 00:00:00",
    )

    result = export_metrics.load_metrics(metrics_path, events_db)

    assert result["equity"] == 999.0


def test_load_latest_metrics_prefers_cycle_metrics(tmp_path: Path) -> None:
    events_db = tmp_path / "events.db"
    _create_events_db(events_db)

    _insert_event(
        events_db,
        "metric",
        {"metrics": {"event_type": "cycle_metrics", "equity": 10.0}},
        "2025-01-01 00:00:00",
    )
    _insert_event(
        events_db,
        "metric",
        {"metrics": {"event_type": "account_snapshot", "equity": 1.0}},
        "2025-01-01 00:01:00",
    )

    result = export_metrics.load_latest_metrics(events_db)

    assert result["equity"] == 10.0


def test_load_latest_event_from_db_normalizes_payload(tmp_path: Path) -> None:
    events_db = tmp_path / "events.db"
    _create_events_db(events_db)

    _insert_event(
        events_db,
        "order_preview",
        {"data": {"foo": "bar"}},
        "2025-01-01 00:00:00",
    )

    event = export_metrics._load_latest_event_from_db(events_db, "order_preview")

    assert event["foo"] == "bar"
    assert event["type"] == "order_preview"
    assert event["timestamp"] == "2025-01-01 00:00:00"


def test_load_latest_event_falls_back_to_file(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "order_preview", "timestamp": "2025-01-01T00:00:00Z"},
        {"event_type": "order_preview", "timestamp": "2025-01-02T00:00:00Z"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    event = export_metrics.load_latest_event(events_path, "order_preview")

    assert event["timestamp"] == "2025-01-02T00:00:00Z"


def test_count_events_uses_database_when_available(tmp_path: Path) -> None:
    events_db = tmp_path / "events.db"
    _create_events_db(events_db)

    _insert_event(events_db, "preview_reject", {}, "2025-01-01 00:00:00")
    _insert_event(events_db, "preview_reject", {}, "2025-01-01 00:01:00")
    _insert_event(events_db, "preview_reject", {}, "2025-01-01 00:02:00")

    count = export_metrics.count_events(tmp_path / "events.jsonl", "preview_reject", events_db)

    assert count == 3


def test_extract_limit_remaining_reads_nested_balances() -> None:
    limits = {"trading_limits": {"remaining_balances": [{"amount": "12.5"}]}}

    remaining = export_metrics._extract_limit_remaining(limits)

    assert remaining == 12.5


def test_render_prometheus_includes_key_metrics(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "preview_reject"},
        {"event_type": "preview_reject"},
        {"event_type": "trade"},
        {"event_type": "trade"},
        {"event_type": "trade_gate_blocked"},
        {"event_type": "websocket_reconnect"},
        {"event_type": "stale_mark_detected"},
        {"event_type": "unfilled_order_alert"},
        {"event_type": "circuit_breaker_triggered"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    metrics = {
        "timestamp": "2025-01-01T00:00:00Z",
        "equity": 1000.0,
        "open_orders": 2,
        "system": {
            "cpu_percent": 12.5,
            "memory_percent": 33.0,
            "disk_percent": 55.0,
            "disk_used_gb": 90.0,
            "network_sent_mb": 1.2,
            "network_recv_mb": 3.4,
        },
        "account_snapshot": {
            "fee_schedule": {"tier": "pro"},
            "limits": {"remaining": "50"},
        },
        "positions": {"BTC-USD": {"exposure_usd": "123.45"}},
        "pnl": {
            "by_symbol": {"BTC-USD": "5.5"},
            "total": "5.5",
            "realized": "2.0",
            "unrealized": "3.0",
            "funding": "0.1",
        },
        "cycle_latency_ms": 250.0,
    }

    output = export_metrics.render_prometheus(metrics, events_path, events_db=None)
    prefix = export_metrics.METRIC_PREFIX
    expected_epoch = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()

    assert f"{prefix}_metrics_timestamp_seconds {expected_epoch}" in output
    assert f"{prefix}_equity 1000.0" in output
    assert f"{prefix}_open_orders 2" in output
    assert f'{prefix}_fee_tier{{tier="pro"}} 1' in output
    assert f"{prefix}_account_limit_remaining 50.0" in output
    assert f"{prefix}_order_preview_failures_total 2" in output
    assert f"{prefix}_trades_executed_total 2" in output
    assert f"{prefix}_trades_blocked_total 1" in output
    assert f"{prefix}_circuit_breaker_triggered 1" in output
    assert f'{prefix}_symbol_exposure_usd{{symbol="BTC_USD"}} 123.45' in output
    assert f'{prefix}_symbol_pnl_usd{{symbol="BTC_USD"}} 5.5' in output


def test_render_prometheus_prefers_trade_counters_when_available(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "trade"},
        {"event_type": "trade"},
        {"event_type": "trade_gate_blocked"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    metrics = {
        "timestamp": "2025-01-01T00:00:00Z",
        "counters": {
            "gpt_trader_trades_executed_total": 7,
            "gpt_trader_trades_blocked_total": 5,
        },
    }

    output = export_metrics.render_prometheus(metrics, events_path, events_db=None)
    prefix = export_metrics.METRIC_PREFIX

    # Should use counters (covers kill-switch blocks), not event counts.
    assert f"{prefix}_trades_executed_total 7" in output
    assert f"{prefix}_trades_blocked_total 5" in output
