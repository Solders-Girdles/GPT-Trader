#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily/period risk report from events.jsonl")
    p.add_argument("--events", help="Path to events.jsonl", default=None)
    p.add_argument("--date", help="End date (YYYY-MM-DD)", default=None)
    p.add_argument("--days", type=int, help="Lookback days (default 1)", default=1)
    p.add_argument("--format", choices=["json", "html"], default="json")
    p.add_argument("--output", help="Output file (default: stdout)", default=None)
    return p.parse_args()


def load_events(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return out


def find_default_events_path() -> Optional[Path]:
    # Prefer per-profile data paths; fallback to managed
    data_root = Path("data/perps_bot")
    if data_root.exists():
        for p in data_root.glob("*/events.jsonl"):
            return p
    managed = Path("results/managed/events.jsonl")
    return managed if managed.exists() else None


def in_range(ts: str, start: datetime, end: datetime) -> bool:
    try:
        t = datetime.fromisoformat(ts)
        return start <= t <= end
    except Exception:
        return False


def summarize(events: List[Dict[str, Any]], start: datetime, end: datetime) -> Dict[str, Any]:
    cb_counts = Counter()
    ws_counts = Counter()
    liq = Counter()
    order_stats = Counter()
    reasons = Counter()
    latest_risk: Dict[str, Any] = {}
    vol_levels: Dict[str, float] = {}

    for e in events:
        if not in_range(e.get("time") or e.get("timestamp") or "", start, end):
            continue
        etype = e.get("type")

        # Circuit breaker events (risk events logged via event_type)
        if (etype == "metric" and e.get("event_type") == "volatility_circuit_breaker") or etype == "volatility_circuit_breaker":
            act = e.get("action") or e.get("metrics", {}).get("action")
            cb_counts[act or "unknown"] += 1
            try:
                sym = e.get("symbol")
                vol = float(e.get("rolling_volatility") or e.get("volatility") or 0)
                if sym:
                    vol_levels[sym] = vol
            except Exception:
                pass

        # WS health events
        if etype in {"ws_gap_detected", "ws_fallback_rest"}:
            ws_counts[etype] += 1

        # Liquidation events
        if etype == "metric" and e.get("event_type") in {"liquidation_buffer_breach"}:
            liq[e.get("event_type")] += 1

        # Order stats
        if etype in {"order_success", "order_failed", "order_rejected"}:
            order_stats[etype] += 1
            if etype == "order_rejected":
                reason = str(e.get("reason") or "")[:80]
                reasons[reason] += 1

        # Risk snapshots
        if etype == "metric" and e.get("bot_id") == "risk_engine":
            latest_risk = {
                "equity": e.get("equity"),
                "total_notional": e.get("total_notional"),
                "exposure_pct": e.get("exposure_pct"),
                "max_leverage": e.get("max_leverage"),
                "daily_pnl": e.get("daily_pnl"),
                "daily_pnl_pct": e.get("daily_pnl_pct"),
                "reduce_only_mode": e.get("reduce_only_mode"),
                "kill_switch_enabled": e.get("kill_switch_enabled"),
            }

    # Approximate uptime if possible
    uptime_pct = None
    total_fallbacks = ws_counts.get("ws_fallback_rest", 0)
    if total_fallbacks == 0:
        uptime_pct = 100.0

    summary = {
        "volatility_circuit_breakers": {
            "warnings": cb_counts.get("warning", 0),
            "reduce_only": cb_counts.get("reduce_only", 0),
            "kill_switch": cb_counts.get("kill_switch", 0),
        },
        "websocket_health": {
            "gaps_detected": ws_counts.get("ws_gap_detected", 0),
            "fallback_events": ws_counts.get("ws_fallback_rest", 0),
            "uptime_pct": uptime_pct,
        },
        "liquidation_events": {
            "buffer_breaches": liq.get("liquidation_buffer_breach", 0),
            "projection_rejections": reasons.get("Projected liquidation buffer", 0),
        },
        "order_stats": {
            "attempted": order_stats.get("order_success", 0) + order_stats.get("order_failed", 0) + order_stats.get("order_rejected", 0),
            "successful": order_stats.get("order_success", 0),
            "failed": order_stats.get("order_failed", 0),
            "success_rate": round(100.0 * (order_stats.get("order_success", 0) / max(1, (order_stats.get("order_success", 0) + order_stats.get("order_failed", 0) + order_stats.get("order_rejected", 0)))), 2),
        },
        "risk_snapshot": latest_risk,
        "volatility_levels": vol_levels,
    }
    return summary


def to_html(report: Dict[str, Any]) -> str:
    import html
    def h(s: Any) -> str:
        return html.escape(str(s))
    body = ["<html><head><title>Risk Report</title></head><body>"]
    body.append("<h1>Risk Report</h1>")
    for section, payload in report.items():
        body.append(f"<h2>{h(section)}</h2>")
        body.append("<pre>")
        body.append(h(json.dumps(payload, indent=2)))
        body.append("</pre>")
    body.append("</body></html>")
    return "\n".join(body)


def main() -> None:
    args = parse_args()
    end = datetime.fromisoformat(args.date) if args.date else datetime.utcnow()
    start = end - timedelta(days=max(1, args.days))

    events_path = Path(args.events) if args.events else find_default_events_path()
    if not events_path or not events_path.exists():
        raise SystemExit("events.jsonl not found; specify --events or generate events first")

    events = load_events(events_path)
    report = {"date": end.date().isoformat(), "summary": summarize(events, start, end)}

    if args.format == "json":
        out = json.dumps(report, indent=2)
    else:
        out = to_html(report)

    if args.output:
        Path(args.output).write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

