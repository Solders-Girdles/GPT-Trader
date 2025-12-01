"""Apply command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import OptimizationStorage
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize apply"


def register(subparsers: Any) -> None:
    """Register the apply subcommand."""
    parser = subparsers.add_parser(
        "apply",
        help="Apply optimized parameters to a config file",
        description="Apply optimized parameters from a study to create a new config profile.",
    )

    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default="latest",
        help="Run ID to apply parameters from (default: latest)",
    )

    # For AI agent envelope output
    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text for YAML config, json for envelope",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Output YAML file path (default: stdout)",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="optimized",
        help="Profile name for the generated config (default: optimized)",
    )

    parser.add_argument(
        "--base-config",
        type=Path,
        metavar="PATH",
        help="Base configuration file to merge with",
    )

    parser.add_argument(
        "--strategy-only",
        action="store_true",
        help="Only include strategy parameters (exclude risk/simulation)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show parameters without writing file",
    )

    parser.set_defaults(handler=execute, subcommand="apply")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the apply command."""
    storage = OptimizationStorage()
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

    if not run.best_parameters:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.NO_BEST_PARAMS,
                message=f"Run {run_id} has no best parameters",
                details={"run_id": run_id},
            )
        logger.error(f"Run {run_id} has no best parameters")
        return 1

    run_data = run.to_dict()

    # Load base config if provided
    base_config: dict[str, Any] = {}
    if args.base_config:
        if not args.base_config.exists():
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.FILE_NOT_FOUND,
                    message=f"Base config file not found: {args.base_config}",
                    details={"path": str(args.base_config)},
                )
            logger.error(f"Base config file not found: {args.base_config}")
            return 1
        try:
            with open(args.base_config) as f:
                base_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.CONFIG_INVALID,
                    message=f"Invalid YAML in base config: {e}",
                    details={"path": str(args.base_config)},
                )
            logger.error(f"Invalid YAML in base config: {e}")
            return 1

    # Build output configuration
    output_config = _build_output_config(
        run_data,
        base_config,
        profile_name=args.profile,
        strategy_only=args.strategy_only,
    )

    # Generate YAML output
    yaml_output = yaml.dump(output_config, default_flow_style=False, sort_keys=False)

    # Handle dry run
    if args.dry_run:
        if output_format == "json":
            return CliResponse.success_response(
                command=COMMAND_NAME,
                data={
                    "config": output_config,
                    "yaml_preview": yaml_output,
                    "run_id": run_id,
                    "study_name": run.study_name,
                    "objective_value": run.best_objective_value,
                },
                was_noop=True,
            )
        print("# Dry run - configuration preview")
        print(f"# From run: {run_id}")
        print(f"# Study: {run.study_name}")
        print(f"# Best objective value: {run.best_objective_value}")
        print()
        print(yaml_output)
        return 0

    # Write output to file
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(f"# Generated from optimization run: {run_id}\n")
                f.write(f"# Study: {run.study_name}\n")
                f.write(f"# Best objective value: {run.best_objective_value}\n")
                f.write("\n")
                f.write(yaml_output)

            if output_format == "json":
                return CliResponse.success_response(
                    command=COMMAND_NAME,
                    data={
                        "written_to": str(args.output),
                        "config": output_config,
                        "run_id": run_id,
                    },
                )
            print(f"Configuration written to: {args.output}")
            return 0
        except Exception as e:
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.OPERATION_FAILED,
                    message=f"Failed to write output file: {e}",
                    details={"path": str(args.output)},
                )
            logger.error(f"Failed to write output file: {e}")
            return 1

    # Output to stdout
    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "config": output_config,
                "yaml_content": yaml_output,
                "run_id": run_id,
            },
        )

    print(yaml_output)
    return 0


def _build_output_config(
    run_data: dict[str, Any],
    base_config: dict[str, Any],
    profile_name: str,
    strategy_only: bool,
) -> dict[str, Any]:
    """
    Build output configuration from optimization results.

    Args:
        run_data: Optimization run data
        base_config: Base configuration to merge with
        profile_name: Profile name for the config
        strategy_only: Whether to include only strategy parameters

    Returns:
        Configuration dictionary
    """
    best_params = run_data.get("best_parameters", {})

    # Categorize parameters
    strategy_params = {}
    risk_params = {}
    simulation_params = {}

    # Known risk parameter names
    risk_param_names = {
        "target_leverage",
        "max_leverage",
        "position_fraction",
        "max_position_size",
        "stop_loss_pct",
        "take_profit_pct",
        "max_drawdown_pct",
        "reduce_only_threshold",
    }

    # Known simulation parameter names
    simulation_param_names = {
        "fee_tier",
        "slippage_bps",
        "spread_impact_pct",
    }

    for name, value in best_params.items():
        if name in risk_param_names:
            risk_params[name] = value
        elif name in simulation_param_names:
            simulation_params[name] = value
        else:
            strategy_params[name] = value

    # Build output structure
    output = dict(base_config)

    # Add profile metadata
    output["profile"] = profile_name
    output["optimization_source"] = {
        "run_id": run_data.get("run_id"),
        "study_name": run_data.get("study_name"),
        "objective_value": run_data.get("best_objective_value"),
    }

    # Add strategy parameters
    if "strategy" not in output:
        output["strategy"] = {}
    output["strategy"].update(strategy_params)

    # Add risk parameters unless strategy-only mode
    if not strategy_only:
        if "risk" not in output:
            output["risk"] = {}
        output["risk"].update(risk_params)

        if simulation_params:
            if "simulation" not in output:
                output["simulation"] = {}
            output["simulation"].update(simulation_params)

    return output
