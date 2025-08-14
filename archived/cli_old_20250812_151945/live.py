from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

from bot.config import get_config
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.live.strategy_selector import SelectionMethod
from bot.portfolio.optimizer import OptimizationMethod
from rich.panel import Panel

from .cli_utils import CLITheme, console


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "live",
        help="Run live orchestrator (strategy selection, optimization, monitoring)",
        description="""
        Start the Production Orchestrator to run autonomous/semi-automated cycles.
        Prints transition smoothness and, if enabled, slippage cost estimates per selection cycle.
        """,
    )

    # Symbols and mode
    p.add_argument(
        "--symbols", required=True, help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in OrchestrationMode],
        default=OrchestrationMode.SEMI_AUTOMATED.value,
        help="Orchestration mode (default: semi_automated)",
    )

    # Intervals
    p.add_argument(
        "--rebalance-interval",
        type=int,
        default=1800,
        help="Rebalance interval in seconds (default: 1800)",
    )
    p.add_argument(
        "--risk-check-interval",
        type=int,
        default=300,
        help="Risk check interval in seconds (default: 300)",
    )
    p.add_argument(
        "--performance-check-interval",
        type=int,
        default=600,
        help="Performance check interval in seconds (default: 600)",
    )

    # Selection and optimization
    p.add_argument(
        "--max-strategies", type=int, default=5, help="Maximum strategies to select (default: 5)"
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum strategy confidence (default: 0.7)",
    )
    p.add_argument(
        "--selection-method",
        choices=[m.value for m in SelectionMethod],
        default=SelectionMethod.HYBRID.value,
        help="Selection method (default: hybrid)",
    )
    p.add_argument(
        "--optimization-method",
        choices=[m.value for m in OptimizationMethod],
        default=OptimizationMethod.SHARPE_MAXIMIZATION.value,
        help="Optimization method (default: sharpe_maximization)",
    )
    p.add_argument(
        "--max-position-weight",
        type=float,
        default=0.4,
        help="Max weight per strategy (default: 0.4)",
    )
    p.add_argument(
        "--target-volatility", type=float, default=0.15, help="Target volatility (default: 0.15)"
    )

    # Turnover/cost knobs
    p.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="Transaction cost in bps (default: 0)",
    )
    p.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Max turnover (L1) per rebalance (optional)",
    )
    p.add_argument(
        "--transition-smoothness-threshold",
        type=float,
        default=None,
        help="If set, send a warning alert when transition smoothness falls below this threshold (0-1)",
    )

    # Slippage hooks
    p.add_argument(
        "--enable-slippage-estimation",
        action="store_true",
        help="Enable slippage estimation during selection cycles",
    )
    p.add_argument(
        "--assumed-portfolio-value",
        type=float,
        default=100000.0,
        help="Assumed portfolio value for estimation (default: 100000)",
    )

    # Run options
    p.add_argument(
        "--print-interval", type=int, default=30, help="Seconds between status prints (default: 30)"
    )

    p.set_defaults(func=_handle_live)
    return p


async def _run_and_print(orchestrator: ProductionOrchestrator, print_interval: int) -> None:
    async def printer() -> None:
        last_seen = 0
        while True:
            try:
                await asyncio.sleep(print_interval)
                ops = orchestrator.get_operation_history("strategy_selection")
                if len(ops) > last_seen:
                    new_ops = ops[last_seen:]
                    last_seen = len(ops)
                    for op in new_ops:
                        data = op.get("data", {})
                        smooth = data.get("transition_smoothness")
                        slip = data.get("slippage_cost_estimate")
                        ts = op.get("timestamp", datetime.now())
                        console.print(
                            CLITheme.info(
                                f"[{ts:%Y-%m-%d %H:%M:%S}] Selection: n={data.get('n_strategies','?')} "
                                f"smoothness={smooth:.3f}"
                                + (
                                    f" slippage_estimate={slip:.2f}"
                                    if isinstance(slip, int | float)
                                    else ""
                                )
                            )
                        )
            except asyncio.CancelledError:
                return
            except Exception:
                # Keep printing loop resilient
                continue

    # Run orchestrator and printer concurrently
    await asyncio.gather(orchestrator.start(), printer())


def _handle_live(args: argparse.Namespace) -> None:
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

    # Parse symbols
    symbols: list[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        console.print(CLITheme.error("No valid symbols provided"))
        sys.exit(1)

    # Display configuration
    console.print(
        Panel(
            f"[bold]Live Orchestrator Configuration[/bold]\n"
            f"Mode: {args.mode}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Rebalance: {args.rebalance_interval}s, Risk: {args.risk_check_interval}s, Perf: {args.performance_check_interval}s\n"
            f"Max strategies: {args.max_strategies}, Min confidence: {args.min_confidence}\n"
            f"Optimization: {args.optimization_method}, Max weight: {args.max_position_weight}, Target vol: {args.target_volatility}\n"
            f"Txn cost (bps): {args.transaction_cost_bps}, Max turnover: {args.max_turnover}\n"
            f"Slippage: {'ON' if args.enable_slippage_estimation else 'OFF'} (assumed value: {args.assumed_portfolio_value:,.0f})",
            title="[bold cyan]Starting Live Orchestrator",
            border_style="cyan",
        )
    )

    # Broker
    broker = AlpacaPaperBroker(
        api_key=app_config.alpaca.api_key_id,
        secret_key=app_config.alpaca.api_secret_key,
        base_url=app_config.alpaca.paper_base_url,
    )

    # Config
    config = OrchestratorConfig(
        mode=OrchestrationMode(args.mode),
        rebalance_interval=args.rebalance_interval,
        risk_check_interval=args.risk_check_interval,
        performance_check_interval=args.performance_check_interval,
        max_strategies=args.max_strategies,
        min_strategy_confidence=args.min_confidence,
        selection_method=SelectionMethod(args.selection_method),
        optimization_method=OptimizationMethod(args.optimization_method),
        max_position_weight=args.max_position_weight,
        target_volatility=args.target_volatility,
        transaction_cost_bps=args.transaction_cost_bps,
        max_turnover=args.max_turnover,
        transition_smoothness_alert_threshold=args.transition_smoothness_threshold,
        enable_slippage_estimation=bool(args.enable_slippage_estimation),
        assumed_portfolio_value=args.assumed_portfolio_value,
        background_enabled=True,
    )

    orchestrator = ProductionOrchestrator(
        config=config,
        broker=broker,
        knowledge_base=None,  # In live usage, this would be initialized with a real KB service
        symbols=symbols,
    )

    console.print(CLITheme.success("Live orchestrator initialized. Press Ctrl+C to stop."))

    try:
        asyncio.run(_run_and_print(orchestrator, args.print_interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Stopping orchestrator...[/yellow]")
        try:
            asyncio.run(orchestrator.stop())
        except Exception:
            pass
        console.print(CLITheme.success("Orchestrator stopped."))
