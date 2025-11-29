"""View command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.commands.optimize.formatters import (
    format_run_summary_text,
    format_trials_csv,
    format_trials_text,
)
from gpt_trader.cli.options import add_output_options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize view"


def register(subparsers: Any) -> None:
    """Register the view subcommand."""
    parser = subparsers.add_parser(
        "view",
        help="View results of a specific optimization run",
        description="View detailed results of an optimization study run.",
    )

    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default="latest",
        help="Run ID to view (default: latest)",
    )

    add_output_options(parser, include_quiet=False)

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        metavar="N",
        help="Show top N trials (default: 10)",
    )

    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Show all parameters for each trial",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, not trial details",
    )

    parser.set_defaults(handler=execute, subcommand="view")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the view command."""
    storage = OptimizationStorage()
    warnings: list[str] = []
    output_format = getattr(args, "output_format", "text")

    # Resolve run ID
    run_id = args.run_id
    if run_id == "latest":
        runs = storage.list_runs()
        if not runs:
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.RUN_NOT_FOUND,
                    message="No optimization runs found",
                )
            logger.error("No optimization runs found")
            return 1
        run_id = runs[0]["run_id"]

    # Load the run
    run = storage.load_run(run_id)
    if run is None:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.RUN_NOT_FOUND,
                message=f"Run not found: {run_id}",
                details={"run_id": run_id},
            )
        logger.error(f"Run not found: {run_id}")
        return 1

    run_data = run.to_dict()

    # Add warnings for potential issues
    if run_data.get("feasible_trials", 0) == 0:
        warnings.append("No feasible trials in this run")

    # Format output
    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=run_data,
            warnings=warnings,
        )

    # Handle CSV format (legacy, not using CliResponse)
    if output_format == "csv":
        trials = run_data.get("trials", [])
        print(format_trials_csv(trials))
        return 0

    # Text format - use existing formatters
    print(format_run_summary_text(run_data))

    if not args.summary_only:
        trials = run_data.get("trials", [])
        if trials:
            print()
            print(
                format_trials_text(
                    trials,
                    limit=args.trials,
                    show_all_params=args.show_params,
                )
            )

    for warning in warnings:
        print(f"Note: {warning}")

    return 0
