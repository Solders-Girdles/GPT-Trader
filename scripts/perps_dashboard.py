#!/usr/bin/env python3
"""
Coinbase Trader Metrics Dashboard

Surfaces Coinbase Trader metrics emitted via EventStore and health.json.
Uses 'rich' library for a professional TUI.

Usage:
  python scripts/perps_dashboard.py --profile dev --refresh 1 --window-min 5
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
from typing import Deque

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from gpt_trader.config.path_registry import RUNTIME_DATA_DIR

console = Console()


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


def summarize(events: Deque[dict], window: timedelta) -> dict:
    now = datetime.now().astimezone()
    cutoff = now - window
    success = 0
    failed = 0
    drift = 0
    pos_drift = 0

    recent_events = []

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

        recent_events.append(evt)

    total = success + failed
    acceptance = (success / total * 100.0) if total > 0 else 0.0

    return {
        "success": success,
        "failed": failed,
        "total": total,
        "acceptance_rate": acceptance,
        "drift_events": drift,
        "position_drift_events": pos_drift,
        "recent_events": list(events)[
            -15:
        ],  # Get absolute last 15 events regardless of window for log
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


def make_header(profile: str, window_min: int) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        f"[b]🚀 Coinbase Trader Dashboard[/b]",
        f"[dim]Profile:[/dim] [cyan]{profile}[/cyan] | [dim]Window:[/dim] [cyan]{window_min}m[/cyan] | [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
    )
    return Panel(grid, style="white on blue")


def make_health_panel(health: dict, metrics: dict) -> Panel:
    status_color = "green" if health.get("ok") else "red"
    status_icon = "✅" if health.get("ok") else "❌"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row(
        "Status",
        f"[{status_color}]{status_icon} {health.get('message', 'Unknown')}[/{status_color}]",
    )
    if health.get("error"):
        table.add_row("Error", f"[red]{health.get('error')}[/red]")
    table.add_row("Last Update", f"[dim]{health.get('timestamp', '')}[/dim]")

    # System Metrics
    sys_metrics = metrics.get("system", {})
    if sys_metrics:
        table.add_row("", "")
        table.add_row("[u]System Resources[/u]", "")

        cpu = sys_metrics.get("cpu_percent", 0)
        mem_pct = sys_metrics.get("memory_percent", 0)
        mem_mb = sys_metrics.get("memory_used_mb", 0)

        cpu_color = "green" if cpu < 50 else "yellow" if cpu < 80 else "red"
        mem_color = "green" if mem_pct < 50 else "yellow" if mem_pct < 80 else "red"

        table.add_row("CPU", f"[{cpu_color}]{cpu:.1f}%[/{cpu_color}]")
        table.add_row("Memory", f"[{mem_color}]{mem_pct:.1f}% ({mem_mb:.0f} MB)[/{mem_color}]")

    return Panel(table, title="System Health", border_style=status_color)


def make_orders_panel(summary: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")

    total = summary["total"]
    success = summary["success"]
    failed = summary["failed"]
    rate = summary["acceptance_rate"]

    table.add_row("Total Orders", str(total))
    table.add_row("Successful", f"[green]{success}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]")

    rate_color = "green" if rate > 90 else "yellow" if rate > 50 else "red"
    table.add_row("Acceptance Rate", f"[{rate_color}]{rate:.1f}%[/{rate_color}]")

    table.add_row("", "")
    table.add_row("[u]Reconciliation[/u]", "")

    drift = summary["drift_events"]
    pos_drift = summary["position_drift_events"]

    drift_style = "green" if drift == 0 else "red"
    pos_drift_style = "green" if pos_drift == 0 else "red"

    table.add_row("Order Drift", f"[{drift_style}]{drift}[/{drift_style}]")
    table.add_row("Position Drift", f"[{pos_drift_style}]{pos_drift}[/{pos_drift_style}]")

    return Panel(table, title="Trading Metrics", border_style="cyan")


def make_events_panel(events: list[dict]) -> Panel:
    table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    table.add_column("Time", style="dim", width=12)
    table.add_column("Type", style="bold", width=15)
    table.add_column("Details")

    for e in reversed(events):  # Show newest first
        t = parse_time(e.get("time"))
        t_str = t.strftime("%H:%M:%S") if t else ""

        etype = e.get("type", "")
        details = ""
        style = "white"

        sym = e.get("symbol") or e.get("product_id") or ""

        if etype == "order_success":
            style = "green"
            details = (
                f"{e.get('side','').upper()} {sym} {e.get('quantity','')} @ {e.get('price', 'MKT')}"
            )
        elif etype == "order_failed":
            style = "red"
            details = f"{sym} {e.get('reason', '')}"
        elif "drift" in etype:
            style = "yellow"
            details = f"local={e.get('local_count')} exch={e.get('exchange_count')}"
        elif etype == "alert":
            style = "red bold"
            details = e.get("message", "")
        else:
            details = str(e)

        table.add_row(t_str, Text(etype, style=style), details)

    return Panel(table, title="Recent Events", border_style="blue")


def main():
    parser = argparse.ArgumentParser(description="Coinbase Trader Metrics Dashboard")
    parser.add_argument("--profile", choices=["dev", "demo", "prod", "canary"], default="dev")
    parser.add_argument("--refresh", type=int, default=1, help="Refresh interval seconds")
    parser.add_argument("--window-min", type=int, default=5, help="Sliding window in minutes")
    args = parser.parse_args()

    # Load config and container to resolve paths consistently
    from gpt_trader.app.container import create_application_container
    from gpt_trader.app.config import BotConfig

    config = BotConfig(profile=args.profile)
    container = create_application_container(config)

    # Use container's runtime paths
    base_dir = container.runtime_paths.event_store_root
    events_path = base_dir / "events.jsonl"
    health_path = base_dir / "health.json"
    metrics_path = base_dir / "metrics.json"

    window = timedelta(minutes=args.window_min)

    layout = Layout()
    layout.split(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=15))
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    with Live(layout, refresh_per_second=1) as live:
        try:
            while True:
                events = load_events(events_path)
                summary = summarize(events, window)
                health = load_health(health_path)
                metrics = load_metrics(metrics_path)

                layout["header"].update(make_header(args.profile, args.window_min))
                layout["left"].update(make_health_panel(health, metrics))
                layout["right"].update(make_orders_panel(summary))
                layout["footer"].update(make_events_panel(summary["recent_events"]))

                time.sleep(args.refresh)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    sys.exit(main())
