from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from bot.config import get_config
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.intelligence.metrics_registry import MetricsRegistry
from bot.monitor.performance_monitor import (
    AlertConfig,
    PerformanceThresholds,
    run_performance_monitor,
)
from rich.panel import Panel

from .cli_utils import CLITheme, console


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "monitor",
        help="Monitor deployed trading strategies",
        description="""
        Monitor the performance of your deployed trading strategies and receive alerts.

        Examples:
            # Basic monitoring
            gpt-trader monitor --min-sharpe 0.8 --max-drawdown 0.12

            # Monitoring with webhook alerts
            gpt-trader monitor --min-sharpe 1.0 --max-drawdown 0.10 \\
                --webhook-url https://hooks.slack.com/... --alert-cooldown 12
        """,
    )

    # Performance thresholds
    threshold_group = p.add_argument_group("Performance Thresholds")
    threshold_group.add_argument(
        "--min-sharpe",
        type=float,
        default=0.8,
        metavar="RATIO",
        help="Minimum Sharpe ratio (default: 0.8)",
    )
    threshold_group.add_argument(
        "--max-drawdown",
        type=float,
        default=0.12,
        metavar="PCT",
        help="Maximum drawdown percentage (default: 0.12)",
    )
    threshold_group.add_argument(
        "--min-cagr",
        type=float,
        default=0.08,
        metavar="PCT",
        help="Minimum CAGR percentage (default: 0.08)",
    )

    # Alert configuration
    alert_group = p.add_argument_group("Alert Configuration")
    alert_group.add_argument(
        "--webhook-url", metavar="URL", help="Webhook URL for sending alerts (e.g., Slack, Discord)"
    )

    # One-shot summary options
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single monitoring cycle and exit (prints summary)",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact status summary (selection metrics, turnover stats) before exit (only with --once)",
    )
    p.add_argument(
        "--audit-summary",
        action="store_true",
        help="Print a compact audit summary (selection_change/rebalance/trade_blocked counts) from logs/audit",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output the one-shot monitoring summary as JSON (use with --once)",
    )
    p.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Write the one-shot monitoring summary to a file (JSON if --json, else text)",
    )
    alert_group.add_argument(
        "--alert-cooldown",
        type=int,
        default=24,
        metavar="HOURS",
        help="Alert cooldown period in hours (default: 24)",
    )

    p.set_defaults(func=_handle_enhanced)
    return p


def _handle_enhanced(args: argparse.Namespace) -> None:
    """Enhanced monitoring handler with better UX."""

    # Load application configuration
    app_config = get_config()

    # Validate Alpaca credentials
    if not app_config.alpaca.api_key_id or not app_config.alpaca.api_secret_key:
        console.print(
            CLITheme.error(
                "Alpaca credentials not found. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
            )
        )
        sys.exit(1)

    # Display configuration
    console.print(
        Panel(
            f"[bold]Monitoring Configuration[/bold]\n"
            f"Min Sharpe: {args.min_sharpe}\n"
            f"Max Drawdown: {args.max_drawdown:.1%}\n"
            f"Min CAGR: {args.min_cagr:.1%}\n"
            f"Alert Cooldown: {args.alert_cooldown}h\n"
            f"Webhook: {'Enabled' if args.webhook_url else 'Disabled'}",
            title="[bold cyan]Starting Performance Monitor",
            border_style="cyan",
        )
    )

    # Create thresholds and alert config
    thresholds = PerformanceThresholds(
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        min_cagr=args.min_cagr,
    )
    alert_cfg = AlertConfig(
        webhook_enabled=bool(args.webhook_url),
        webhook_url=args.webhook_url or None,
        alert_cooldown_hours=args.alert_cooldown,
    )

    # Create broker
    broker = AlpacaPaperBroker(
        api_key=app_config.alpaca.api_key_id,
        secret_key=app_config.alpaca.api_secret_key,
        base_url=app_config.alpaca.paper_base_url,
    )

    # One-shot mode: run a single cycle and optionally print a compact summary
    if getattr(args, "once", False):
        from bot.monitor.performance_monitor import PerformanceMonitor

        async def _run_once() -> None:
            mon = PerformanceMonitor(broker, thresholds, alert_cfg)
            await mon._monitoring_cycle()  # run one cycle
            summary = mon.get_performance_summary()
            # If JSON output requested
            if getattr(args, "json", False):
                rendered = json.dumps(summary, indent=2, default=str)
                if getattr(args, "output", None):
                    Path(args.output).write_text(rendered)
                console.print(rendered)
                return
            # Pretty printed compact summary
            if getattr(args, "print_summary", False):
                sel = summary.get("selection_metrics", {}) or {}
                turn = summary.get("turnover_stats", {}) or {}
                text = (
                    f"Selection metrics: top_k={sel.get('top_k_accuracy', 0):.3f}, "
                    f"rank_corr={sel.get('rank_correlation', 0):.3f}, "
                    f"regret={sel.get('regret', 0):.3f}\n"
                    f"Turnover: mean={turn.get('mean', 0.0):.4f}, p95={turn.get('p95', 0.0):.4f} (n={turn.get('count', 0)})"
                )
                if getattr(args, "output", None):
                    Path(args.output).write_text(text)
                console.print(Panel(text, title="Monitoring Summary", border_style="green"))
            # Optional: audit summary from JSONL
            if getattr(args, "audit_summary", False):
                try:
                    ops_path = Path("logs/audit/operations.jsonl")
                    counts = {"selection_change": 0, "rebalance": 0, "trade_blocked": 0}
                    latest: dict[str, dict] = {k: None for k in counts.keys()}  # type: ignore[assignment]
                    last_blocked: dict | None = None
                    if ops_path.exists():
                        with ops_path.open() as f:
                            for line in f:
                                import json as _json

                                rec = _json.loads(line)
                                op = rec.get("operation")
                                if op in counts:
                                    counts[op] += 1
                                    latest[op] = rec
                                    if op == "trade_blocked":
                                        last_blocked = rec
                    # Build details line for last blocked rebalance (if any)
                    details = ""
                    if last_blocked:
                        data = last_blocked.get("data", {})
                        reason = data.get("reason", "unknown")
                        l1 = data.get("turnover_l1")
                        cap = data.get("turnover_cap")
                        ts = last_blocked.get("timestamp", "")
                        if isinstance(l1, int | float) and isinstance(cap, int | float):
                            details = f"Last blocked ({ts}): reason={reason}, turnover_l1={float(l1):.3f}, cap={float(cap):.3f}"
                        else:
                            details = f"Last blocked ({ts}): reason={reason}"
                    text = (
                        f"Audit counts: selection_change={counts['selection_change']}, "
                        f"rebalance={counts['rebalance']}, trade_blocked={counts['trade_blocked']}"
                        + (f"\n{details}" if details else "")
                    )
                    console.print(
                        Panel(
                            text,
                            title="Audit Summary (logs/audit)",
                            border_style="blue",
                        )
                    )
                except Exception:
                    console.print(
                        Panel("No audit data available.", title="Audit Summary", border_style="red")
                    )

        asyncio.run(_run_once())
        return

    console.print(CLITheme.success("Performance monitor started. Press Ctrl+C to stop."))

    try:
        # Start monitor (continuous)
        asyncio.run(run_performance_monitor(broker, thresholds, alert_cfg))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Stopping performance monitor...[/yellow]")
        console.print(CLITheme.success("Performance monitor stopped."))

    # Optional: compare recent portfolio versions from metrics registry
    try:
        reg = MetricsRegistry(Path("logs/metrics"))
        versions = reg.list_versions() if hasattr(reg, "list_versions") else []
        if versions:
            latest = versions[-3:]
            comparison = reg.compare_versions(latest)
            console.print(
                Panel(
                    f"Recent portfolio metrics comparison:\n{comparison}", title="Metrics Registry"
                )
            )
    except Exception:
        pass

    # Optional: one-off status snapshot for turnover and selection metrics
    try:
        from bot.monitor.performance_monitor import PerformanceMonitor

        # Create a temporary monitor to read summary if the orchestrator is not running
        # In typical live mode, this is driven by orchestrator, but here we only show schema
        console.print(
            Panel(
                "Tip: In live mode, use orchestrator summaries to view turnover_stats (mean/p95) and selection metrics.",
                title="Monitoring Snapshot",
            )
        )
    except Exception:
        pass


# Keep the original handler for backward compatibility
def _handle(args: argparse.Namespace) -> None:
    """Original monitoring handler for backward compatibility."""
    _handle_enhanced(args)
