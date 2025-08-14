from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot.backtest.engine_portfolio import run_backtest as run_backtest_port
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.base import Strategy
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
from rich.panel import Panel
from rich.table import Table

from .cli_utils import CLITheme, console
from .shared import (
    _compose_bt_basename,
    _ensure_run_dir,
    _guesstimate_universe_label,
    _parse_date,
    _wait_for_summary_since,
)


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the backtest subcommand with enhanced help."""

    p = subparsers.add_parser(
        "backtest",
        help="Run strategy backtests with historical data",
        description="""
        Run comprehensive backtests on your trading strategies using historical market data.

        Examples:
            # Basic backtest on single symbol
            gpt-trader backtest --symbol AAPL --start 2023-01-01 --end 2023-12-31

            # Backtest with custom parameters
            gpt-trader backtest --symbol-list universe.csv --start 2023-01-01 --end 2023-12-31 \\
                --donchian 55 --atr 20 --risk-pct 1.0 --max-positions 10

            # Backtest with regime filter
            gpt-trader backtest --symbol SPY --start 2023-01-01 --end 2023-12-31 \\
                --regime on --regime-window 200
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Strategy parameters
    strategy_group = p.add_argument_group("Strategy Configuration")
    strategy_group.add_argument(
        "--strategy",
        default="trend_breakout",
        choices=["demo_ma", "trend_breakout"],
        help="Trading strategy to test (default: trend_breakout)",
    )
    strategy_group.add_argument(
        "--donchian",
        type=int,
        default=55,
        metavar="DAYS",
        help="Donchian channel lookback period (default: 55)",
    )
    strategy_group.add_argument(
        "--atr",
        type=int,
        default=20,
        metavar="DAYS",
        help="ATR period for volatility calculation (default: 20)",
    )
    strategy_group.add_argument(
        "--atr-k",
        type=float,
        default=2.0,
        metavar="MULT",
        help="ATR multiplier for stop-loss (default: 2.0)",
    )

    # Data parameters
    data_group = p.add_argument_group("Data Selection")
    data_group.add_argument(
        "--symbol", metavar="TICKER", help="Single symbol to backtest (e.g., AAPL)"
    )
    data_group.add_argument(
        "--symbol-list", metavar="FILE", help="CSV file containing symbols to test"
    )
    data_group.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD", help="Start date for backtest"
    )
    data_group.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD", help="End date for backtest"
    )

    # Risk management
    risk_group = p.add_argument_group("Risk Management")
    risk_group.add_argument(
        "--risk-pct",
        type=float,
        default=0.5,
        metavar="PCT",
        help="Risk per trade as %% of portfolio (default: 0.5)",
    )
    risk_group.add_argument(
        "--max-positions",
        type=int,
        default=10,
        metavar="N",
        help="Maximum concurrent positions (default: 10)",
    )
    risk_group.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        metavar="BPS",
        help="Transaction cost in basis points (default: 0)",
    )

    # Regime filter
    regime_group = p.add_argument_group("Regime Filter")
    regime_group.add_argument(
        "--regime", choices=["on", "off"], default="off", help="Enable regime filter (default: off)"
    )
    regime_group.add_argument(
        "--regime-symbol",
        default="SPY",
        metavar="TICKER",
        help="Symbol for regime detection (default: SPY)",
    )
    regime_group.add_argument(
        "--regime-window",
        type=int,
        default=200,
        metavar="DAYS",
        help="Moving average window for regime (default: 200)",
    )

    # Execution parameters
    exec_group = p.add_argument_group("Execution Settings")
    exec_group.add_argument(
        "--cadence",
        choices=["daily", "weekly"],
        default="daily",
        help="Rebalancing frequency (default: daily)",
    )
    exec_group.add_argument(
        "--entry-confirm",
        type=int,
        default=1,
        metavar="DAYS",
        help="Days to confirm entry signal (default: 1)",
    )
    exec_group.add_argument(
        "--cooldown",
        type=int,
        default=0,
        metavar="DAYS",
        help="Cooldown period between trades (default: 0)",
    )
    exec_group.add_argument(
        "--exit-mode",
        choices=["signal", "stop"],
        default="signal",
        help="Exit strategy mode (default: signal)",
    )
    exec_group.add_argument(
        "--min-rebalance-pct",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Minimum position change to trigger rebalance (default: 0)",
    )

    # Output options
    output_group = p.add_argument_group("Output Options")
    output_group.add_argument(
        "--run-tag", default="", metavar="TAG", help="Tag for experiment folder naming"
    )
    output_group.add_argument(
        "--run-dir", default="", metavar="DIR", help="Override output directory"
    )
    output_group.add_argument("--no-plot", action="store_true", help="Disable chart generation")
    output_group.add_argument("--debug", action="store_true", help="Enable debug output")

    p.set_defaults(func=_handle_enhanced)
    return p


def _handle_enhanced(args: argparse.Namespace) -> None:
    """Enhanced backtest handler with better UX."""

    # Validate inputs
    if not args.symbol and not args.symbol_list:
        console.print(CLITheme.error("Either --symbol or --symbol-list must be provided"))
        sys.exit(1)

    # Display backtest configuration
    console.print(
        Panel(
            f"[bold]Backtest Configuration[/bold]\n"
            f"Strategy: {args.strategy}\n"
            f"Period: {args.start} to {args.end}\n"
            f"Risk per trade: {args.risk_pct}%\n"
            f"Max positions: {args.max_positions}",
            title="[bold cyan]Starting Backtest",
            border_style="cyan",
        )
    )

    # Setup strategy
    if args.strategy == "demo_ma":
        strategy: Strategy = DemoMAStrategy()
    else:
        strategy = TrendBreakoutStrategy(
            TrendBreakoutParams(
                donchian_lookback=args.donchian,
                atr_period=args.atr,
                atr_k=args.atr_k,
            )
        )

    # Setup risk rules
    rules = PortfolioRules(
        per_trade_risk_pct=args.risk_pct / 100.0,
        atr_k=args.atr_k,
        max_positions=args.max_positions,
        cost_bps=args.cost_bps,
    )

    # Setup output directory
    bt_run_dir: Path | None = None
    if getattr(args, "run_dir", "") or getattr(args, "run_tag", ""):
        bt_run_dir = _ensure_run_dir(
            args.strategy,
            run_tag=getattr(args, "run_tag", ""),
            run_dir=getattr(args, "run_dir", ""),
        )
        console.print(CLITheme.info(f"Output directory: {bt_run_dir}"))

    # Track existing files
    outdir = Path("data/backtests")
    outdir.mkdir(parents=True, exist_ok=True)
    before = {str(p) for p in outdir.glob("PORT_*_summary.csv")}

    # Run backtest with progress indicator
    with console.status("[bold green]Running backtest...") as status:
        try:
            run_backtest_port(
                symbol=args.symbol,
                symbol_list_csv=args.symbol_list,
                start=_parse_date(args.start),
                end=_parse_date(args.end),
                strategy=strategy,
                rules=rules,
                regime_on=(args.regime == "on"),
                regime_symbol=args.regime_symbol,
                regime_window=args.regime_window,
                debug=args.debug,
                exit_mode=args.exit_mode,
                cadence=args.cadence,
                cooldown=args.cooldown,
                entry_confirm=args.entry_confirm,
                min_rebalance_pct=args.min_rebalance_pct,
                strict_mode=getattr(args, "strict_mode", True),
                make_plot=not args.no_plot,
                show_progress=True,
                write_portfolio_csv=True,
                write_trades_csv=True,
                write_summary_csv=True,
                progress_desc="Backtesting",
            )
            status.update("[bold green]Backtest complete!")
        except Exception as e:
            console.print(CLITheme.error(f"Backtest failed: {e}"))
            raise

    # Process results
    newest = _wait_for_summary_since(before, timeout=3.0, poll=0.05)
    if newest and newest.exists():
        # Display results summary
        display_backtest_results(newest)

        # Save to experiment directory if specified
        if bt_run_dir:
            sym_label = _guesstimate_universe_label(
                getattr(args, "symbol", None), getattr(args, "symbol_list", None)
            )
            base_name = _compose_bt_basename(
                args.strategy,
                sym_label,
                str(args.start),
                str(args.end),
                getattr(args, "donchian", None),
                getattr(args, "atr", None),
                getattr(args, "atr_k", None),
                getattr(args, "risk_pct", None),
                str(getattr(args, "cadence", "daily")),
                (getattr(args, "regime", "off") == "on"),
                getattr(args, "regime_window", None),
            )
            from .shared import _mirror_backtest_triplet_from_summary

            _mirror_backtest_triplet_from_summary(newest, bt_run_dir / "backtests", base_name)
            console.print(CLITheme.success(f"Results saved to {bt_run_dir}"))
    else:
        console.print(CLITheme.warning("No summary file generated"))


def display_backtest_results(summary_path: Path) -> None:
    """Display backtest results in a formatted table."""
    import pandas as pd

    try:
        # Read the summary file - it's in key-value format
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        # Parse key-value pairs (taking the last occurrence if duplicates exist)
        metrics_dict = {}
        for line in lines:
            if ',' in line:
                key, value = line.strip().split(',', 1)
                try:
                    metrics_dict[key] = float(value)
                except ValueError:
                    metrics_dict[key] = value

        # Create results table
        table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")

        # Key metrics to display with correct field names from the CSV
        metrics = [
            ("Total Return", "total_return", lambda x: f"{x*100:.2f}%"),
            ("CAGR", "cagr", lambda x: f"{x*100:.2f}%"),
            ("Sharpe Ratio", "sharpe", lambda x: f"{x:.3f}"),
            ("Max Drawdown", "max_drawdown", lambda x: f"{x*100:.2f}%"),
            ("Volatility", "vol", lambda x: f"{x*100:.2f}%"),
            ("Total Costs", "total_costs", lambda x: f"${x:.2f}"),
        ]

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for label, key, formatter in metrics:
            if key in metrics_dict:
                value = metrics_dict[key]
                table.add_row(label, formatter(value))

        console.print(table)

    except Exception as e:
        console.print(CLITheme.warning(f"Could not display results: {e}"))


# Keep the original handler for backward compatibility
def _handle(args: argparse.Namespace) -> None:
    """Original backtest handler for backward compatibility."""
    _handle_enhanced(args)
