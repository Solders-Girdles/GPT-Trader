"""Simple HTTP exporter for GPT-Trader metrics.

Serves Prometheus-compatible metrics and a JSON view of the latest
`cycle_metrics` and `account_snapshot` data so dashboards/alerting stacks
can consume telemetry without touching internal modules.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "var" / "data" / "perps_bot" / "prod" / "metrics.json"
DEFAULT_EVENTS = REPO_ROOT / "var" / "data" / "perps_bot" / "prod" / "events.jsonl"

from flask import Flask, Response, jsonify


app = Flask(__name__)


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


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


def count_events(event_path: Path, event_type: str) -> int:
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


def load_latest_event(event_path: Path, event_type: str) -> dict[str, Any]:
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


def render_prometheus(metrics: dict[str, Any], events_path: Path) -> str:
    lines: list[str] = []
    system = metrics.get("system", {})
    timestamp = metrics.get("timestamp")
    lines.append("# HELP gpt_trader_metrics_timestamp_seconds Timestamp of last cycle metrics")
    lines.append("# TYPE gpt_trader_metrics_timestamp_seconds gauge")
    if isinstance(timestamp, str):
        try:
            import datetime as _dt

            ts = _dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            epoch = ts.timestamp()
        except Exception:
            epoch = 0.0
    else:
        epoch = 0.0
    lines.append(f"gpt_trader_metrics_timestamp_seconds {epoch}")

    lines.append("# HELP gpt_trader_equity Account equity in USD")
    lines.append("# TYPE gpt_trader_equity gauge")
    try:
        lines.append(f"gpt_trader_equity {float(metrics.get('equity', 0))}")
    except Exception:
        lines.append("gpt_trader_equity 0")

    lines.append("# HELP gpt_trader_open_orders Number of open orders")
    lines.append("# TYPE gpt_trader_open_orders gauge")
    lines.append(f"gpt_trader_open_orders {int(metrics.get('open_orders', 0))}")

    lines.append("# HELP gpt_trader_cpu_percent Process CPU percent")
    lines.append("# TYPE gpt_trader_cpu_percent gauge")
    lines.append(f"gpt_trader_cpu_percent {system.get('cpu_percent', 0)}")

    lines.append("# HELP gpt_trader_memory_percent Process memory percent")
    lines.append("# TYPE gpt_trader_memory_percent gauge")
    lines.append(f"gpt_trader_memory_percent {system.get('memory_percent', 0)}")

    snapshot = metrics.get("account_snapshot") or {}
    fee_schedule = snapshot.get("fee_schedule") or {}
    limits = snapshot.get("limits") or {}
    fee_tier = fee_schedule.get("tier", "unknown")
    lines.append("# HELP gpt_trader_fee_tier Account fee tier (label)")
    lines.append("# TYPE gpt_trader_fee_tier gauge")
    lines.append(f'gpt_trader_fee_tier{{tier="{fee_tier}"}} 1')

    remaining_limit = _extract_limit_remaining(limits)
    lines.append("# HELP gpt_trader_account_limit_remaining Remaining trading limit in USD")
    lines.append("# TYPE gpt_trader_account_limit_remaining gauge")
    lines.append(f"gpt_trader_account_limit_remaining {remaining_limit}")

    preview_failures = count_events(events_path, "preview_reject")
    lines.append(
        "# HELP gpt_trader_order_preview_failures_total Total number of order preview failures observed"
    )
    lines.append("# TYPE gpt_trader_order_preview_failures_total counter")
    lines.append(f"gpt_trader_order_preview_failures_total {preview_failures}")

    return "\n".join(lines) + "\n"


def create_app(metrics_path: Path, events_path: Path) -> Flask:
    @app.route("/metrics")
    def metrics_route():
        metrics = load_metrics(metrics_path)
        return Response(render_prometheus(metrics, events_path), mimetype="text/plain")

    @app.route("/metrics.json")
    def metrics_json():
        metrics = load_metrics(metrics_path)
        metrics["latest_order_preview"] = load_latest_event(events_path, "order_preview")
        metrics["latest_account_snapshot"] = load_latest_event(events_path, "account_snapshot")
        return jsonify(metrics)

    return app


def main():
    parser = argparse.ArgumentParser(description="GPT-Trader metrics exporter")
    parser.add_argument("--metrics-file", default=str(DEFAULT_METRICS), help="Path to metrics.json")
    parser.add_argument("--events-file", default=str(DEFAULT_EVENTS), help="Path to events log")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    events_path = Path(args.events_file)
    app = create_app(metrics_path, events_path)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
