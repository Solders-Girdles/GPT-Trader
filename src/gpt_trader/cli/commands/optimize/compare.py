"""Compare command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.commands.optimize.formatters import (
    COMPARISON_METRICS,
    format_comparison_text,
)
from gpt_trader.cli.commands.optimize.config_loader import resolve_optimize_preset_inheritance
from gpt_trader.cli.options import add_output_options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

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

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Run ID to treat as the baseline for delta calculations (default: first run)",
    )

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
        run_data["config"] = resolve_optimize_preset_inheritance(run_data.get("config", {}))
        run_summary = {
            "run_id": run_data["run_id"],
            "study_name": run_data["study_name"],
            "best_objective_value": run_data.get("best_objective_value"),
            "total_trials": run_data.get("total_trials", 0),
            "feasible_trials": run_data.get("feasible_trials", 0),
            "best_parameters": run_data.get("best_parameters"),
            "started_at": run_data.get("started_at"),
            "completed_at": run_data.get("completed_at"),
        }
        runs.append(run_summary)

        # Add warning if run has no feasible trials
        if run_summary.get("feasible_trials", 0) == 0:
            warnings.append(f"Run {run_id} has no feasible trials")
    # Determine baseline run (default to first provided run)
    baseline_id = getattr(args, "baseline", None) or args.run_ids[0]
    run_ids = [run["run_id"] for run in runs]
    if baseline_id not in run_ids:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.INVALID_ARGUMENT,
                message=f"Baseline run {baseline_id} is not part of the comparison set",
                details={"baseline_run_id": baseline_id},
            )
        logger.error(f"Baseline run {baseline_id} is not part of the comparison set")
        return 1

    comparison_data = _build_comparison_payload(runs, baseline_id)

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=comparison_data,
            warnings=warnings,
        )

    # Text format
    print(format_comparison_text(comparison_data))
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


def _build_comparison_payload(runs: list[dict[str, Any]], baseline_run_id: str) -> dict[str, Any]:
    """Build comparison payload including baseline metadata and metric matrix."""
    baseline_run = _find_run(runs, baseline_run_id)

    matrix = []
    baseline_metrics: dict[str, Any] = {}
    for metric_key, metric_label in COMPARISON_METRICS:
        baseline_value = baseline_run.get(metric_key)
        baseline_metrics[metric_key] = baseline_value
        row_values = []
        for run in runs:
            value = run.get(metric_key)
            delta = _calculate_delta(value, baseline_value)
            row_values.append(
                {
                    "run_id": run["run_id"],
                    "study_name": run["study_name"],
                    "value": value,
                    "delta": delta,
                }
            )
        matrix.append({"metric": metric_key, "label": metric_label, "values": row_values})

    baseline_metadata = {
        "run_id": baseline_run["run_id"],
        "study_name": baseline_run["study_name"],
        "started_at": baseline_run.get("started_at"),
        "completed_at": baseline_run.get("completed_at"),
        "metrics": baseline_metrics,
    }

    return {
        "runs": runs,
        "baseline_run": baseline_metadata,
        "baseline_run_id": baseline_run_id,
        "matrix": matrix,
        "best_run": _find_best_run(runs),
    }


def _find_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Find run data by run ID."""
    for run in runs:
        if run["run_id"] == run_id:
            return run
    return runs[0]


def _calculate_delta(value: float | int | None, baseline: float | int | None) -> float | int | None:
    """Return difference between value and baseline if both present."""
    if value is None or baseline is None:
        return None
    delta = value - baseline
    if isinstance(value, int) and isinstance(baseline, int):
        return int(delta)
    return float(delta)
