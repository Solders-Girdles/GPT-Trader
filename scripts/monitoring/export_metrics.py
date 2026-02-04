"""Simple HTTP exporter for Coinbase Trader metrics.

Serves Prometheus-compatible metrics and a JSON view of the latest
`cycle_metrics` and `account_snapshot` data so dashboards/alerting stacks
can consume telemetry without touching internal modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Flask

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "prod"
DEFAULT_RUNTIME_ROOT = REPO_ROOT
DEFAULT_METRICS = DEFAULT_RUNTIME_ROOT / "runtime_data" / DEFAULT_PROFILE / "metrics.json"
DEFAULT_EVENTS_DB = DEFAULT_RUNTIME_ROOT / "runtime_data" / DEFAULT_PROFILE / "events.db"
DEFAULT_EVENTS = DEFAULT_RUNTIME_ROOT / "runtime_data" / DEFAULT_PROFILE / "events.jsonl"
METRIC_PREFIX = os.getenv("COINBASE_TRADER_METRIC_PREFIX", "coinbase_trader")


def load_metrics(metrics_path: Path, events_db: Path | None = None) -> dict[str, Any]:
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            return {}
    if events_db is not None:
        metrics = load_latest_metrics(events_db)
        if metrics:
            return metrics
    return {}


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_event(
    event_type: str, payload: dict[str, Any], *, fallback_timestamp: Any | None
) -> dict[str, Any] | None:
    if not event_type:
        event_type = "unknown"
    if "data" in payload and isinstance(payload["data"], dict) and len(payload) == 1:
        payload = payload["data"]
    event = dict(payload)
    event["type"] = event_type
    ts = payload.get("timestamp")
    if not ts and fallback_timestamp is not None:
        ts = fallback_timestamp
    if ts:
        event["timestamp"] = ts
    return event


def load_latest_metrics(events_db: Path) -> dict[str, Any]:
    if not events_db.exists():
        return {}
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
    except Exception:
        return {}
    finally:
        if connection is not None:
            connection.close()


def _load_latest_event_from_db(events_db: Path, event_type: str) -> dict[str, Any]:
    if not events_db.exists():
        return {}
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(events_db))
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            SELECT timestamp, event_type, payload
            FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (event_type,),
        )
        row = cursor.fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(row["payload"])
        except Exception:
            return {"timestamp": row["timestamp"], "type": event_type}
        if not isinstance(payload, dict):
            return {"timestamp": row["timestamp"], "type": event_type}
        normalized = _normalize_event(event_type, payload, fallback_timestamp=row["timestamp"])
        return normalized or {"timestamp": row["timestamp"], "type": event_type}
    except Exception:
        return {}
    finally:
        if connection is not None:
            connection.close()


def _count_events_in_db(events_db: Path, event_type: str) -> int:
    if not events_db.exists():
        return 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(events_db))
        cursor = connection.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE event_type = ?
            """,
            (event_type,),
        )
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0
    except Exception:
        return 0
    finally:
        if connection is not None:
            connection.close()


def _extract_limit_remaining(limits: dict[str, Any]) -> float:
    if not isinstance(limits, dict):
        return 0.0
    for key in (
        "remaining",
        "remaining_balance",
        "available",
        "max_order",
        "available_trading_balance",
    ):
        if key in limits and limits[key] is not None:
            return _parse_float(limits[key])
    trading = limits.get("trading_limits")
    if isinstance(trading, dict):
        for key in ("max_order", "remaining", "available"):
            if key in trading and trading[key] is not None:
                return _parse_float(trading[key])
        remaining = trading.get("remaining_balances")
        if isinstance(remaining, list) and remaining:
            amount = remaining[0].get("amount")
            if amount is not None:
                return _parse_float(amount)
    return 0.0


def count_events(event_path: Path, event_type: str, events_db: Path | None = None) -> int:
    if events_db is not None and events_db.exists():
        count = _count_events_in_db(events_db, event_type)
        return count
    if not event_path.exists():
        return 0
    count = 0
    try:
        with event_path.open("r") as f:
            for line in f:
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if evt.get("event_type") == event_type or evt.get("type") == event_type:
                    count += 1
    except Exception:
        return 0
    return count


def load_latest_event(
    event_path: Path,
    event_type: str,
    *,
    events_db: Path | None = None,
) -> dict[str, Any]:
    if events_db is not None and events_db.exists():
        event = _load_latest_event_from_db(events_db, event_type)
        if event:
            return event
    if not event_path.exists():
        return {}
    try:
        with event_path.open("r") as f:
            for line in reversed(f.readlines()):
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if evt.get("event_type") == event_type or evt.get("type") == event_type:
                    return evt
    except Exception:
        pass
    return {}


def render_prometheus(metrics: dict[str, Any], events_path: Path, events_db: Path | None) -> str:
    lines: list[str] = []
    system = metrics.get("system", {})
    timestamp = metrics.get("timestamp")
    if not timestamp and events_db is not None:
        latest_metric = _load_latest_event_from_db(events_db, "metric")
        timestamp = latest_metric.get("timestamp")

    def metric(name: str) -> str:
        return f"{METRIC_PREFIX}_{name}"

    lines.append(f"# HELP {metric('metrics_timestamp_seconds')} Timestamp of last cycle metrics")
    lines.append(f"# TYPE {metric('metrics_timestamp_seconds')} gauge")
    if isinstance(timestamp, str):
        try:
            import datetime as _dt

            ts = _dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            epoch = ts.timestamp()
        except Exception:
            epoch = 0.0
    else:
        epoch = 0.0
    lines.append(f"{metric('metrics_timestamp_seconds')} {epoch}")

    lines.append(f"# HELP {metric('equity')} Account equity in USD")
    lines.append(f"# TYPE {metric('equity')} gauge")
    try:
        lines.append(f"{metric('equity')} {float(metrics.get('equity', 0))}")
    except Exception:
        lines.append(f"{metric('equity')} 0")

    lines.append(f"# HELP {metric('open_orders')} Number of open orders")
    lines.append(f"# TYPE {metric('open_orders')} gauge")
    lines.append(f"{metric('open_orders')} {int(metrics.get('open_orders', 0))}")

    lines.append(f"# HELP {metric('cpu_percent')} Process CPU percent")
    lines.append(f"# TYPE {metric('cpu_percent')} gauge")
    lines.append(f"{metric('cpu_percent')} {system.get('cpu_percent', 0)}")

    lines.append(f"# HELP {metric('memory_percent')} Process memory percent")
    lines.append(f"# TYPE {metric('memory_percent')} gauge")
    lines.append(f"{metric('memory_percent')} {system.get('memory_percent', 0)}")

    lines.append(f"# HELP {metric('disk_percent')} Process disk usage percent")
    lines.append(f"# TYPE {metric('disk_percent')} gauge")
    lines.append(f"{metric('disk_percent')} {system.get('disk_percent', 0)}")

    lines.append(f"# HELP {metric('disk_used_gigabytes')} Disk usage (GB)")
    lines.append(f"# TYPE {metric('disk_used_gigabytes')} gauge")
    lines.append(
        f"{metric('disk_used_gigabytes')} {system.get('disk_used_gb', system.get('disk_gb', 0))}"
    )

    lines.append(f"# HELP {metric('network_sent_megabytes')} Total MB sent")
    lines.append(f"# TYPE {metric('network_sent_megabytes')} counter")
    lines.append(f"{metric('network_sent_megabytes')} {system.get('network_sent_mb', 0)}")

    lines.append(f"# HELP {metric('network_received_megabytes')} Total MB received")
    lines.append(f"# TYPE {metric('network_received_megabytes')} counter")
    lines.append(f"{metric('network_received_megabytes')} {system.get('network_recv_mb', 0)}")

    snapshot = metrics.get("account_snapshot") or {}
    fee_schedule = snapshot.get("fee_schedule") or {}
    limits = snapshot.get("limits") or {}
    fee_tier = fee_schedule.get("tier", "unknown")
    lines.append(f"# HELP {metric('fee_tier')} Account fee tier (label)")
    lines.append(f"# TYPE {metric('fee_tier')} gauge")
    lines.append(f'{metric("fee_tier")}{{tier="{fee_tier}"}} 1')

    remaining_limit = _extract_limit_remaining(limits)
    lines.append(f"# HELP {metric('account_limit_remaining')} Remaining trading limit in USD")
    lines.append(f"# TYPE {metric('account_limit_remaining')} gauge")
    lines.append(f"{metric('account_limit_remaining')} {remaining_limit}")

    preview_failures = count_events(events_path, "preview_reject", events_db)
    lines.append(
        f"# HELP {metric('order_preview_failures_total')} Total number of order preview failures observed"
    )
    lines.append(f"# TYPE {metric('order_preview_failures_total')} counter")
    lines.append(f"{metric('order_preview_failures_total')} {preview_failures}")

    trades_executed = count_events(events_path, "trade", events_db)
    lines.append(
        f"# HELP {metric('trades_executed_total')} Total number of trades executed successfully"
    )
    lines.append(f"# TYPE {metric('trades_executed_total')} counter")
    lines.append(f"{metric('trades_executed_total')} {trades_executed}")

    trades_blocked = count_events(events_path, "trade_gate_blocked", events_db)
    lines.append(
        f"# HELP {metric('trades_blocked_total')} Total number of trades blocked by guards"
    )
    lines.append(f"# TYPE {metric('trades_blocked_total')} counter")
    lines.append(f"{metric('trades_blocked_total')} {trades_blocked}")

    # Circuit breaker state
    circuit_breaker_triggered = 0
    latest_cb = load_latest_event(events_path, "circuit_breaker_triggered", events_db=events_db)
    if latest_cb:
        circuit_breaker_triggered = 1
    lines.append(
        f"# HELP {metric('circuit_breaker_triggered')} Circuit breaker triggered (1=active, 0=normal)"
    )
    lines.append(f"# TYPE {metric('circuit_breaker_triggered')} gauge")
    lines.append(f"{metric('circuit_breaker_triggered')} {circuit_breaker_triggered}")

    # Cycle latency
    cycle_latency = metrics.get("cycle_latency_ms", 0)
    lines.append(f"# HELP {metric('cycle_latency_ms')} Trading cycle latency in milliseconds")
    lines.append(f"# TYPE {metric('cycle_latency_ms')} gauge")
    lines.append(f"{metric('cycle_latency_ms')} {cycle_latency}")

    # Per-symbol exposure
    positions = metrics.get("positions", {})
    if isinstance(positions, dict):
        lines.append(f"# HELP {metric('symbol_exposure_usd')} Per-symbol exposure in USD")
        lines.append(f"# TYPE {metric('symbol_exposure_usd')} gauge")
        for symbol, position_data in positions.items():
            if isinstance(position_data, dict):
                exposure = _parse_float(position_data.get("exposure_usd", 0))
                # Sanitize symbol for prometheus label
                safe_symbol = symbol.replace("-", "_")
                lines.append(
                    f'{metric("symbol_exposure_usd")}{{symbol="{safe_symbol}"}} {exposure}'
                )

    # Per-symbol PnL
    pnl_data = metrics.get("pnl", {})
    if isinstance(pnl_data, dict):
        symbol_pnl = pnl_data.get("by_symbol", {})
        if isinstance(symbol_pnl, dict):
            lines.append(f"# HELP {metric('symbol_pnl_usd')} Per-symbol total PnL in USD")
            lines.append(f"# TYPE {metric('symbol_pnl_usd')} gauge")
            for symbol, pnl_value in symbol_pnl.items():
                pnl = _parse_float(pnl_value)
                safe_symbol = symbol.replace("-", "_")
                lines.append(f'{metric("symbol_pnl_usd")}{{symbol="{safe_symbol}"}} {pnl}')

    # Total PnL metrics
    total_pnl = _parse_float(pnl_data.get("total", 0))
    realized_pnl = _parse_float(pnl_data.get("realized", 0))
    unrealized_pnl = _parse_float(pnl_data.get("unrealized", 0))
    funding_pnl = _parse_float(pnl_data.get("funding", 0))

    lines.append(f"# HELP {metric('total_pnl_usd')} Total PnL in USD")
    lines.append(f"# TYPE {metric('total_pnl_usd')} gauge")
    lines.append(f"{metric('total_pnl_usd')} {total_pnl}")

    lines.append(f"# HELP {metric('realized_pnl_usd')} Realized PnL in USD")
    lines.append(f"# TYPE {metric('realized_pnl_usd')} gauge")
    lines.append(f"{metric('realized_pnl_usd')} {realized_pnl}")

    lines.append(f"# HELP {metric('unrealized_pnl_usd')} Unrealized PnL in USD")
    lines.append(f"# TYPE {metric('unrealized_pnl_usd')} gauge")
    lines.append(f"{metric('unrealized_pnl_usd')} {unrealized_pnl}")

    lines.append(f"# HELP {metric('funding_pnl_usd')} Funding PnL in USD")
    lines.append(f"# TYPE {metric('funding_pnl_usd')} gauge")
    lines.append(f"{metric('funding_pnl_usd')} {funding_pnl}")

    # Health metrics
    ws_reconnects = count_events(events_path, "websocket_reconnect", events_db)
    stale_marks = count_events(events_path, "stale_mark_detected", events_db)
    unfilled_orders = count_events(events_path, "unfilled_order_alert", events_db)

    lines.append(f"# HELP {metric('websocket_reconnects_total')} Total WebSocket reconnections")
    lines.append(f"# TYPE {metric('websocket_reconnects_total')} counter")
    lines.append(f"{metric('websocket_reconnects_total')} {ws_reconnects}")

    lines.append(f"# HELP {metric('stale_marks_total')} Total stale mark detections")
    lines.append(f"# TYPE {metric('stale_marks_total')} counter")
    lines.append(f"{metric('stale_marks_total')} {stale_marks}")

    lines.append(f"# HELP {metric('unfilled_orders_total')} Total unfilled order alerts")
    lines.append(f"# TYPE {metric('unfilled_orders_total')} counter")
    lines.append(f"{metric('unfilled_orders_total')} {unfilled_orders}")

    return "\n".join(lines) + "\n"


def create_app(metrics_path: Path, events_path: Path, events_db: Path | None) -> Flask:
    from flask import Flask, Response, jsonify  # type: ignore[import-not-found]

    app = Flask(__name__)

    @app.route("/metrics")
    def metrics_route():
        metrics = load_metrics(metrics_path, events_db)
        return Response(render_prometheus(metrics, events_path, events_db), mimetype="text/plain")

    @app.route("/metrics.json")
    def metrics_json():
        metrics = load_metrics(metrics_path, events_db)
        metrics["latest_order_preview"] = load_latest_event(
            events_path, "order_preview", events_db=events_db
        )
        metrics["latest_account_snapshot"] = load_latest_event(
            events_path, "account_snapshot", events_db=events_db
        )
        return jsonify(metrics)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Coinbase Trader metrics exporter")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Runtime profile name")
    parser.add_argument(
        "--runtime-root",
        default=str(DEFAULT_RUNTIME_ROOT),
        help="Repo root containing runtime_data/<profile>",
    )
    parser.add_argument("--metrics-file", default=str(DEFAULT_METRICS), help="Path to metrics.json")
    parser.add_argument("--events-db", default=str(DEFAULT_EVENTS_DB), help="Path to events.db")
    parser.add_argument("--events-file", default=str(DEFAULT_EVENTS), help="Path to events log")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    if args.metrics_file == str(DEFAULT_METRICS):
        metrics_path = (
            Path(args.runtime_root).expanduser() / "runtime_data" / args.profile / "metrics.json"
        )
    else:
        metrics_path = Path(args.metrics_file).expanduser()

    if args.events_db == str(DEFAULT_EVENTS_DB):
        events_db = (
            Path(args.runtime_root).expanduser() / "runtime_data" / args.profile / "events.db"
        )
    else:
        events_db = Path(args.events_db).expanduser()

    if args.events_file == str(DEFAULT_EVENTS):
        events_path = (
            Path(args.runtime_root).expanduser() / "runtime_data" / args.profile / "events.jsonl"
        )
    else:
        events_path = Path(args.events_file).expanduser()

    if not events_db.exists():
        events_db = None

    app = create_app(metrics_path, events_path, events_db)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
