"""Export command for optimization CLI."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from gpt_trader.cli.commands.optimize.formatters import format_trials_csv
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, enable_console=True)

COMMAND_NAME = "optimize export"


def register(subparsers: Any) -> None:
    """Register the export subcommand."""
    parser = subparsers.add_parser(
        "export",
        help="Export optimization results",
        description="Export optimization study results to various formats.",
    )

    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default="latest",
        help="Run ID to export (default: latest)",
    )

    # Export has its own format options distinct from envelope format
    parser.add_argument(
        "--export-format",
        dest="export_format",
        type=str,
        choices=["json", "csv", "yaml"],
        default="json",
        help="Export format (default: json)",
    )

    # For AI agent envelope output
    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text for raw export, json for envelope",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Export only best trial parameters",
    )

    parser.add_argument(
        "--include-trials",
        action="store_true",
        help="Include all trial data in export",
    )

    parser.set_defaults(handler=execute, subcommand="export")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the export command."""
    storage = OptimizationStorage()
    output_format = getattr(args, "output_format", "text")
    export_format = getattr(args, "export_format", "json")
    warnings: list[str] = []

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

    # Prepare export data
    if args.best_only:
        export_data = {
            "run_id": run_data["run_id"],
            "study_name": run_data["study_name"],
            "objective_value": run_data.get("best_objective_value"),
            "parameters": run_data.get("best_parameters", {}),
        }
    else:
        export_data = {
            "run_id": run_data["run_id"],
            "study_name": run_data["study_name"],
            "started_at": run_data["started_at"],
            "completed_at": run_data.get("completed_at"),
            "total_trials": run_data["total_trials"],
            "feasible_trials": run_data["feasible_trials"],
            "best_objective_value": run_data.get("best_objective_value"),
            "best_parameters": run_data.get("best_parameters"),
            "config": run_data.get("config"),
        }

        if args.include_trials:
            export_data["trials"] = run_data.get("trials", [])

    # Format output based on export format
    if export_format == "csv":
        trials = run_data.get("trials", [])
        if not trials:
            warnings.append("No trial data available for CSV export")
            exported_content = ""
        else:
            exported_content = format_trials_csv(trials)
    elif export_format == "yaml":
        exported_content = yaml.dump(export_data, default_flow_style=False, sort_keys=False)
    else:  # json
        exported_content = json.dumps(export_data, indent=2)

    # Write output to file if specified
    output_path = args.output
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(exported_content)

            if output_format == "json":
                return CliResponse.success_response(
                    command=COMMAND_NAME,
                    data={
                        "exported_to": str(output_path),
                        "export_format": export_format,
                        "run_id": run_id,
                    },
                    warnings=warnings,
                )
            print(f"Exported to: {output_path}")
            return 0
        except Exception as e:
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.OPERATION_FAILED,
                    message=f"Failed to write output file: {e}",
                    details={"path": str(output_path)},
                )
            logger.error(f"Failed to write output file: {e}")
            return 1

    # Output to stdout
    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "content": exported_content,
                "export_format": export_format,
                "run_id": run_id,
            },
            warnings=warnings,
        )

    print(exported_content)
    return 0
