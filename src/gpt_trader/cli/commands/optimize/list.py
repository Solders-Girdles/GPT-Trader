"""List command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.commands.optimize.formatters import format_run_list_text
from gpt_trader.cli.options import add_output_options
from gpt_trader.cli.response import CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize list"


def register(subparsers: Any) -> None:
    """Register the list subcommand."""
    parser = subparsers.add_parser(
        "list",
        help="List all optimization runs",
        description="List all saved optimization study runs.",
    )

    add_output_options(parser, include_quiet=False)

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of runs to show (default: 20)",
    )

    parser.add_argument(
        "--study",
        type=str,
        metavar="NAME",
        help="Filter by study name",
    )

    parser.set_defaults(handler=execute, subcommand="list")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the list command."""
    storage = OptimizationStorage()
    warnings: list[str] = []

    runs = storage.list_runs()

    # Apply filters
    if args.study:
        runs = [r for r in runs if args.study.lower() in r.get("study_name", "").lower()]

    # Apply limit
    total_count = len(runs)
    runs = runs[: args.limit]
    if total_count > args.limit:
        warnings.append(f"Showing {args.limit} of {total_count} runs")

    # Build response data
    data = {
        "runs": runs,
        "total_count": total_count,
        "shown_count": len(runs),
    }

    output_format = getattr(args, "output_format", "text")

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=data,
            warnings=warnings,
        )

    # Text format - use existing formatter
    if not runs:
        if args.study:
            print(f"No runs found matching study name: {args.study}")
        else:
            print("No optimization runs found.")
            print()
            print("Run 'coinbase-trader optimize run' to start a new optimization study.")
    else:
        print(format_run_list_text(runs))
        for warning in warnings:
            print(f"Note: {warning}")

    return 0
