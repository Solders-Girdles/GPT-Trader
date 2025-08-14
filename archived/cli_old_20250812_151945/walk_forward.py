from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot.optimization.walk_forward_validator import (
    WalkForwardConfig,
    run_walk_forward_validation,
)
from rich.panel import Panel

from .cli_utils import CLITheme, console


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "walk-forward",
        help="Walk-forward validation of trading strategies",
        description="""
        Perform walk-forward validation to test strategy robustness over time.

        Examples:
            # Basic walk-forward validation
            gpt-trader walk-forward --results optimization_results.csv --symbols AAPL,MSFT,GOOGL

            # Custom validation windows
            gpt-trader walk-forward --results results.csv --symbols SPY,QQQ \\
                --train-months 18 --test-months 3 --step-months 3 --min-windows 5
        """,
    )

    # Input parameters
    input_group = p.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--results", required=True, metavar="FILE", help="Path to optimization results CSV file"
    )
    input_group.add_argument(
        "--symbols",
        metavar="SYMBOLS",
        help="Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)",
    )

    # Validation parameters
    validation_group = p.add_argument_group("Validation Parameters")
    validation_group.add_argument(
        "--train-months",
        type=int,
        default=12,
        metavar="MONTHS",
        help="Training period in months (default: 12)",
    )
    validation_group.add_argument(
        "--test-months",
        type=int,
        default=6,
        metavar="MONTHS",
        help="Testing period in months (default: 6)",
    )
    validation_group.add_argument(
        "--step-months",
        type=int,
        default=6,
        metavar="MONTHS",
        help="Step size in months (default: 6)",
    )
    validation_group.add_argument(
        "--min-windows",
        type=int,
        default=3,
        metavar="N",
        help="Minimum number of validation windows (default: 3)",
    )
    validation_group.add_argument(
        "--min-mean-sharpe",
        type=float,
        default=0.5,
        metavar="RATIO",
        help="Minimum mean Sharpe ratio (default: 0.5)",
    )

    # Output parameters
    output_group = p.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output",
        metavar="FILE",
        help="Output path for validated results (default: <input>_wf_validated.csv)",
    )

    p.set_defaults(func=_handle_enhanced)
    return p


def _handle_enhanced(args: argparse.Namespace) -> None:
    """Enhanced walk-forward validation handler with better UX."""

    # Validate input file
    results_path = Path(args.results)
    if not results_path.exists():
        console.print(CLITheme.error(f"Results file not found: {args.results}"))
        sys.exit(1)

    # Parse symbols
    symbols = [
        s.strip().upper() for s in (args.symbols or "AAPL,MSFT,GOOGL").split(",") if s.strip()
    ]

    # Determine output path
    output_path = args.output or args.results.replace(".csv", "_wf_validated.csv")

    # Display configuration
    console.print(
        Panel(
            f"[bold]Walk-Forward Configuration[/bold]\n"
            f"Results file: {args.results}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Train period: {args.train_months} months\n"
            f"Test period: {args.test_months} months\n"
            f"Step size: {args.step_months} months\n"
            f"Min windows: {args.min_windows}\n"
            f"Min mean Sharpe: {args.min_mean_sharpe}\n"
            f"Output: {output_path}",
            title="[bold cyan]Starting Walk-Forward Validation",
            border_style="cyan",
        )
    )

    # Create validation config
    config = WalkForwardConfig(
        symbols=symbols,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        min_windows=args.min_windows,
        min_mean_sharpe=args.min_mean_sharpe,
    )

    # Run walk-forward validation
    with console.status("[bold green]Running walk-forward validation...") as status:
        try:
            run_walk_forward_validation(args.results, config, output_path=output_path)
            status.update("[bold green]Walk-forward validation complete!")
            console.print(CLITheme.success(f"Validation results saved to {output_path}"))
        except Exception as e:
            console.print(CLITheme.error(f"Walk-forward validation failed: {e}"))
            raise


# Keep the original handler for backward compatibility
def _handle(args: argparse.Namespace) -> None:
    """Original walk-forward validation handler for backward compatibility."""
    _handle_enhanced(args)
