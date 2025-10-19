#!/usr/bin/env python3
"""
Coinbase Trader Metrics Dashboard

Surfaces Coinbase Trader metrics emitted via EventStore and health.json.

Shows:
- Order success/failure counts and acceptance rate (sliding window)
- Drift detection frequency (order_drift)
- Bot health status (from health.json)

Usage:
  python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5

Env overrides:
  EVENT_STORE_ROOT: base directory for events/health (defaults to var/data/coinbase_trader/<profile>;
  falls back to the legacy var/data/perps_bot/<profile> if present)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from bot_v2.config.path_registry import RUNTIME_DATA_DIR


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def load_events(path: Path, max_lines: int = 5000) -> Deque[dict]:
    events: Deque[dict] = deque(maxlen=max_lines)
    if not path.exists():
        return events
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                events.append(evt)
    except Exception:
        pass
    return events


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def summarize(events: Deque[dict], window: timedelta) -> dict[str, float | int]:
    now = datetime.utcnow().astimezone()
    cutoff = now - window
    success = 0
    failed = 0
    drift = 0
    pos_drift = 0
    for evt in events:
        ts = parse_time(evt.get("time"))
        if ts is None or ts < cutoff:
            continue
        etype = str(evt.get("type", "")).lower()
        if etype == "order_success":
            success += 1
        elif etype == "order_failed":
            failed += 1
        elif etype == "order_drift":
            drift += 1
        elif etype == "position_drift":
            pos_drift += 1
    total = success + failed
    acceptance = (success / total * 100.0) if total > 0 else 0.0
    return {
        "success": success,
        "failed": failed,
        "total": total,
        "acceptance_rate": acceptance,
        "drift_events": drift,
        "position_drift_events": pos_drift,
    }


def load_health(health_path: Path) -> dict:
    if not health_path.exists():
        return {"ok": False, "message": "health.json not found"}
    try:
        with health_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        return {"ok": False, "message": f"error reading health: {e}"}


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Coinbase Trader Metrics Dashboard")
    parser.add_argument("--profile", choices=["dev", "demo", "prod", "canary"], default="dev")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval seconds")
    parser.add_argument("--window-min", type=int, default=5, help="Sliding window in minutes")
    args = parser.parse_args()

    # Resolve base dir for this profile (aligns with Coinbase Trader EventStore root)
    default_root = RUNTIME_DATA_DIR / "coinbase_trader" / args.profile
    legacy_root = RUNTIME_DATA_DIR / "perps_bot" / args.profile
    if not default_root.exists() and legacy_root.exists():
        default_root = legacy_root

    base_dir = Path(os.getenv("EVENT_STORE_ROOT", str(default_root)))
    events_path = base_dir / "events.jsonl"
    health_path = base_dir / "health.json"
    metrics_path = base_dir / "metrics.json"

    # Load once, then loop
    window = timedelta(minutes=args.window_min)

    try:
        while True:
            clear()
            print("=" * 80)
            print(
                f"🚀 Coinbase Trader Metrics Dashboard  |  Profile: {args.profile}  |  Window: {args.window_min}m"
            )
            print("=" * 80)

            events = load_events(events_path)
            summary = summarize(events, window)
            health = load_health(health_path)
            metrics = load_metrics(metrics_path)

            # Health
            ok_icon = "✅" if health.get("ok") else "❌"
            print("\n🩺 Health")
            print("-" * 40)
            print(
                f"Status: {ok_icon}  |  Message: {health.get('message','')}  |  Error: {health.get('error','')}"
            )
            print(f"Time:   {health.get('timestamp', '')}")

            system_metrics = metrics.get("system") or {}
            if system_metrics:
                print("\n🖥️  System Resources")
                print("-" * 40)
                cpu = system_metrics.get("cpu_percent", 0)
                mem_pct = system_metrics.get("memory_percent", 0)
                mem_mb = system_metrics.get("memory_used_mb", system_metrics.get("memory_mb", 0))
                disk_pct = system_metrics.get("disk_percent", 0)
                disk_gb = system_metrics.get("disk_used_gb", system_metrics.get("disk_gb", 0))
                net_tx = system_metrics.get("network_sent_mb", 0)
                net_rx = system_metrics.get("network_recv_mb", 0)
                threads = system_metrics.get("threads", system_metrics.get("system_threads", 0))
                print(f"CPU:     {float(cpu):5.1f}%")
                print(f"Memory:  {float(mem_pct):5.1f}% ({float(mem_mb):.0f} MB)")
                print(f"Disk:    {float(disk_pct):5.1f}% ({float(disk_gb):.2f} GB used)")
                print(f"Network: sent {float(net_tx):.1f} MB  recv {float(net_rx):.1f} MB")
                print(f"Threads: {int(threads) if str(threads).isdigit() else threads}")

            # Orders
            print(f"\n📊 Orders (last {args.window_min:d} min)")
            print("-" * 40)
            print(f"Total:     {summary['total']}")
            print(f"Successful:{summary['success']}")
            print(f"Failed:    {summary['failed']}")
            print(f"Acceptance:{summary['acceptance_rate']:.1f}%")

            # Drift
            print("\n🔁 Reconciliation")
            print("-" * 40)
            print(f"order_drift events:    {summary['drift_events']}")
            print(f"position_drift events: {summary['position_drift_events']}")

            # Tail last few events for context
            tail = list(events)[-10:]
            if tail:
                print("\n🧾 Recent Events")
                print("-" * 40)
                for e in tail:
                    t = e.get("time", "")
                    et = e.get("type", "")
                    sym = e.get("symbol") or e.get("product_id") or ""
                    msg = ""
                    if et == "order_success":
                        msg = f"{et} {sym} {e.get('side','')} quantity={e.get('quantity','')}"
                    elif et == "order_failed":
                        msg = et
                    elif et == "order_drift":
                        msg = f"{et} local={e.get('local_count')} exch={e.get('exchange_count')}"
                    else:
                        msg = et
                    print(f"{t}  {msg}")

            print("\n" + "=" * 80)
            print(f"Events: {events_path}  |  Health: {health_path}")
            print("Press Ctrl+C to exit")
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    sys.exit(main())
