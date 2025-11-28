"""Compare command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.commands.optimize.formatters import format_comparison_text
from gpt_trader.cli.options import add_output_options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, enable_console=True)

COMMAND_NAME = "optimize compare"


def register(subparsers: Any) -> None:
    """Register the compare subcommand."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare multiple optimization runs",
        description="Compare results across multiple optimization study runs.",
    )

    parser.add_argument(
        "run_ids",
        type=str,
        nargs="+",
        help="Run IDs to compare (at least 2)",
    )

    add_output_options(parser, include_quiet=False)

    parser.set_defaults(handler=execute, subcommand="compare")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the compare command."""
    output_format = getattr(args, "output_format", "text")

    if len(args.run_ids) < 2:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.INSUFFICIENT_RUNS,
                message="At least 2 run IDs are required for comparison",
                details={"provided_count": len(args.run_ids)},
            )
        logger.error("At least 2 run IDs are required for comparison")
        return 1

    storage = OptimizationStorage()
    warnings: list[str] = []

    # Load all runs
    runs = []
    for run_id in args.run_ids:
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
        runs.append(run_data)

        # Add warning if run has no feasible trials
        if run_data.get("feasible_trials", 0) == 0:
            warnings.append(f"Run {run_id} has no feasible trials")

    # Build comparison data
    comparison_data = {
        "runs": [
            {
                "run_id": r["run_id"],
                "study_name": r["study_name"],
                "best_objective_value": r.get("best_objective_value"),
                "total_trials": r["total_trials"],
                "feasible_trials": r["feasible_trials"],
                "best_parameters": r.get("best_parameters"),
            }
            for r in runs
        ],
        "best_run": _find_best_run(runs),
    }

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=comparison_data,
            warnings=warnings,
        )

    # Text format
    print(format_comparison_text(runs))
    for warning in warnings:
        print(f"Note: {warning}")

    return 0


def _find_best_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the run with the best objective value."""
    best = None
    best_value = None
    for run in runs:
        value = run.get("best_objective_value")
        if value is not None and (best_value is None or value > best_value):
            best_value = value
            best = {
                "run_id": run["run_id"],
                "objective_value": value,
            }
    return best
