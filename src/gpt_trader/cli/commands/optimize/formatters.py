"""Output formatters for optimization CLI commands."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any


def format_run_summary_text(run: dict[str, Any]) -> str:
    """
    Format an optimization run summary for human-readable output.

    Args:
        run: Dictionary containing run data

    Returns:
        Formatted text string
    """
    lines = [
        "=" * 80,
        f"{'OPTIMIZATION STUDY: ' + run['study_name']:^80}",
        "=" * 80,
        "",
        f"Run ID: {run['run_id']}",
        f"Started: {_format_datetime(run['started_at'])}",
    ]

    if run.get("completed_at"):
        lines.append(f"Completed: {_format_datetime(run['completed_at'])}")

    lines.extend([
        f"Total Trials: {run['total_trials']}",
        f"Feasible Trials: {run['feasible_trials']}",
        "",
    ])

    if run.get("best_objective_value") is not None:
        lines.extend([
            "-" * 40,
            "BEST TRIAL",
            "-" * 40,
            f"Objective Value: {run['best_objective_value']:.4f}",
            "",
            "Parameters:",
        ])

        if run.get("best_parameters"):
            for key, value in sorted(run["best_parameters"].items()):
                lines.append(f"  {key}: {_format_value(value)}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_run_summary_json(run: dict[str, Any], compact: bool = False) -> str:
    """
    Format an optimization run summary as JSON.

    Args:
        run: Dictionary containing run data
        compact: If True, use compact JSON format

    Returns:
        JSON string
    """
    output = {
        "status": "completed" if run.get("completed_at") else "in_progress",
        "run_id": run["run_id"],
        "study_name": run["study_name"],
        "started_at": run["started_at"],
        "completed_at": run.get("completed_at"),
        "total_trials": run["total_trials"],
        "feasible_trials": run["feasible_trials"],
        "best_trial": None,
    }

    if run.get("best_objective_value") is not None:
        output["best_trial"] = {
            "objective_value": run["best_objective_value"],
            "parameters": run.get("best_parameters"),
        }

    if compact:
        return json.dumps(output, separators=(",", ":"))
    return json.dumps(output, indent=2)


def format_trials_text(
    trials: list[dict[str, Any]],
    limit: int = 10,
    show_all_params: bool = False,
) -> str:
    """
    Format a list of trials for human-readable output.

    Args:
        trials: List of trial dictionaries
        limit: Maximum number of trials to show
        show_all_params: If True, show all parameters

    Returns:
        Formatted text string
    """
    if not trials:
        return "No trials found."

    # Sort by objective value (descending for maximize)
    sorted_trials = sorted(
        trials,
        key=lambda t: t.get("objective_value", float("-inf")),
        reverse=True,
    )[:limit]

    lines = [
        f"Top {len(sorted_trials)} Trials:",
        "-" * 60,
        f"{'#':>4} {'Objective':>12} {'Feasible':>8} {'Duration':>10}",
        "-" * 60,
    ]

    for trial in sorted_trials:
        feasible = "Yes" if trial.get("is_feasible", False) else "No"
        duration = f"{trial.get('duration_seconds', 0):.1f}s"
        obj_val = trial.get("objective_value")
        obj_str = f"{obj_val:.4f}" if obj_val is not None else "N/A"

        lines.append(
            f"{trial.get('trial_number', '?'):>4} {obj_str:>12} {feasible:>8} {duration:>10}"
        )

        if show_all_params and trial.get("parameters"):
            for key, value in sorted(trial["parameters"].items()):
                lines.append(f"      {key}: {_format_value(value)}")

    return "\n".join(lines)


def format_trials_csv(trials: list[dict[str, Any]]) -> str:
    """
    Format trials as CSV.

    Args:
        trials: List of trial dictionaries

    Returns:
        CSV string
    """
    if not trials:
        return ""

    # Collect all parameter keys
    param_keys: set[str] = set()
    for trial in trials:
        if trial.get("parameters"):
            param_keys.update(trial["parameters"].keys())

    param_keys_sorted = sorted(param_keys)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["trial_number", "objective_value", "is_feasible", "duration_seconds"]
    header.extend(param_keys_sorted)
    writer.writerow(header)

    # Rows
    for trial in trials:
        row = [
            trial.get("trial_number", ""),
            trial.get("objective_value", ""),
            trial.get("is_feasible", ""),
            trial.get("duration_seconds", ""),
        ]
        params = trial.get("parameters", {})
        for key in param_keys_sorted:
            row.append(params.get(key, ""))
        writer.writerow(row)

    return output.getvalue()


def format_run_list_text(runs: list[dict[str, Any]]) -> str:
    """
    Format a list of optimization runs.

    Args:
        runs: List of run summaries

    Returns:
        Formatted text string
    """
    if not runs:
        return "No optimization runs found."

    lines = [
        "Optimization Runs:",
        "-" * 70,
        f"{'Run ID':<24} {'Study Name':<20} {'Best Value':>12} {'Date':>12}",
        "-" * 70,
    ]

    for run in runs:
        best_val = run.get("best_value")
        best_str = f"{best_val:.4f}" if best_val is not None else "N/A"
        date_str = _format_date_short(run.get("started_at", ""))
        run_id = run.get("run_id", "")[:22]
        study_name = run.get("study_name", "")[:18]

        lines.append(f"{run_id:<24} {study_name:<20} {best_str:>12} {date_str:>12}")

    return "\n".join(lines)


def format_run_list_json(runs: list[dict[str, Any]], compact: bool = False) -> str:
    """
    Format a list of optimization runs as JSON.

    Args:
        runs: List of run summaries
        compact: If True, use compact JSON format

    Returns:
        JSON string
    """
    if compact:
        return json.dumps({"runs": runs}, separators=(",", ":"))
    return json.dumps({"runs": runs}, indent=2)


def format_comparison_text(runs: list[dict[str, Any]]) -> str:
    """
    Format a comparison of multiple runs.

    Args:
        runs: List of run data dictionaries

    Returns:
        Formatted comparison text
    """
    if not runs:
        return "No runs to compare."

    if len(runs) == 1:
        return format_run_summary_text(runs[0])

    lines = [
        "=" * 80,
        f"{'OPTIMIZATION COMPARISON':^80}",
        "=" * 80,
        "",
    ]

    # Summary table
    lines.extend([
        "-" * 80,
        f"{'Study':<20} {'Objective':>12} {'Trials':>8} {'Feasible':>8} {'Run ID':<24}",
        "-" * 80,
    ])

    for run in runs:
        best_val = run.get("best_objective_value")
        best_str = f"{best_val:.4f}" if best_val is not None else "N/A"
        lines.append(
            f"{run['study_name'][:18]:<20} {best_str:>12} "
            f"{run['total_trials']:>8} {run['feasible_trials']:>8} "
            f"{run['run_id'][:22]:<24}"
        )

    # Parameter comparison if all runs have best parameters
    all_have_params = all(run.get("best_parameters") for run in runs)
    if all_have_params:
        lines.extend(["", "-" * 80, "Best Parameters Comparison:", "-" * 80])

        # Collect all parameter keys
        all_keys: set[str] = set()
        for run in runs:
            all_keys.update(run.get("best_parameters", {}).keys())

        for key in sorted(all_keys):
            values = []
            for run in runs:
                params = run.get("best_parameters", {})
                values.append(_format_value(params.get(key, "N/A")))

            lines.append(f"  {key}:")
            for i, (run, val) in enumerate(zip(runs, values)):
                lines.append(f"    [{run['study_name'][:15]}] {val}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_comparison_json(runs: list[dict[str, Any]], compact: bool = False) -> str:
    """
    Format a comparison of multiple runs as JSON.

    Args:
        runs: List of run data dictionaries
        compact: If True, use compact JSON format

    Returns:
        JSON string
    """
    comparison = {
        "comparison": [
            {
                "run_id": run["run_id"],
                "study_name": run["study_name"],
                "best_objective_value": run.get("best_objective_value"),
                "total_trials": run["total_trials"],
                "feasible_trials": run["feasible_trials"],
                "best_parameters": run.get("best_parameters"),
            }
            for run in runs
        ]
    }

    if compact:
        return json.dumps(comparison, separators=(",", ":"))
    return json.dumps(comparison, indent=2)


def _format_datetime(dt_str: str | None) -> str:
    """Format ISO datetime string for display."""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(dt_str)


def _format_date_short(dt_str: str | None) -> str:
    """Format ISO datetime string as short date."""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(dt_str)[:10]


def _format_value(value: Any) -> str:
    """Format a parameter value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
