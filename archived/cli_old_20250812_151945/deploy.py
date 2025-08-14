from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot.optimization.deployment_pipeline import (
    DeploymentConfig,
    run_deployment_pipeline,
)
from rich.panel import Panel

from .cli_utils import CLITheme, console


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "deploy",
        help="Deploy optimized strategies to live trading",
        description="""
        Deploy your best-performing optimized strategies to live trading.

        Examples:
            # Deploy from optimization results
            gpt-trader deploy --results optimization_results.csv --symbols AAPL,MSFT,GOOGL

            # Deploy with custom filters
            gpt-trader deploy --results results.csv --symbols SPY,QQQ \\
                --min-sharpe 1.5 --max-drawdown 0.10 --min-trades 50
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

    # Filter parameters
    filter_group = p.add_argument_group("Strategy Filters")
    filter_group.add_argument(
        "--min-sharpe",
        type=float,
        default=1.0,
        metavar="RATIO",
        help="Minimum Sharpe ratio (default: 1.0)",
    )
    filter_group.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        metavar="PCT",
        help="Maximum drawdown percentage (default: 0.15)",
    )
    filter_group.add_argument(
        "--min-trades",
        type=int,
        default=20,
        metavar="N",
        help="Minimum number of trades (default: 20)",
    )
    filter_group.add_argument(
        "--max-strategies",
        type=int,
        default=3,
        metavar="N",
        help="Maximum strategies to deploy (default: 3)",
    )
    filter_group.add_argument(
        "--validation-days",
        type=int,
        default=30,
        metavar="DAYS",
        help="Validation period in days (default: 30)",
    )

    # Output parameters
    output_group = p.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        default="data/deployment",
        metavar="DIR",
        help="Output directory for deployment results (default: data/deployment)",
    )

    p.set_defaults(func=_handle_enhanced)
    return p


def _handle_enhanced(args: argparse.Namespace) -> None:
    """Enhanced deployment handler with better UX."""

    # Validate input file
    results_path = Path(args.results)
    if not results_path.exists():
        console.print(CLITheme.error(f"Results file not found: {args.results}"))
        sys.exit(1)

    # Parse symbols
    symbols = [
        s.strip().upper() for s in (args.symbols or "AAPL,MSFT,GOOGL").split(",") if s.strip()
    ]

    # Display configuration
    console.print(
        Panel(
            f"[bold]Deployment Configuration[/bold]\n"
            f"Results file: {args.results}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Min Sharpe: {args.min_sharpe}\n"
            f"Max Drawdown: {args.max_drawdown:.1%}\n"
            f"Min Trades: {args.min_trades}\n"
            f"Max Strategies: {args.max_strategies}",
            title="[bold cyan]Starting Deployment",
            border_style="cyan",
        )
    )

    # Create deployment config
    config = DeploymentConfig(
        symbols=symbols,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        min_trades=args.min_trades,
        max_concurrent_strategies=args.max_strategies,
        validation_period_days=args.validation_days,
    )

    # Run deployment pipeline
    with console.status("[bold green]Running deployment pipeline...") as status:
        try:
            run_deployment_pipeline(args.results, config, output_dir=args.output_dir)
            status.update("[bold green]Deployment complete!")
            console.print(CLITheme.success(f"Deployment results saved to {args.output_dir}"))
        except Exception as e:
            console.print(CLITheme.error(f"Deployment failed: {e}"))
            raise


# Keep the original handler for backward compatibility
def _handle(args: argparse.Namespace) -> None:
    """Original deployment handler for backward compatibility."""
    _handle_enhanced(args)
