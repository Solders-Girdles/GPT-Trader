"""Resume command for optimization CLI."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from typing import Any

from gpt_trader.cli.commands.optimize.config_loader import (
    ConfigValidationError,
    create_objective_from_preset,
)
from gpt_trader.cli.commands.optimize.formatters import format_run_summary_text
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import (
    OptimizationRun,
    OptimizationStorage,
)
from gpt_trader.features.optimize.runner.batch_runner import TrialResult
from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import OptimizationConfig
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize resume"


def register(subparsers: Any) -> None:
    """Register the resume subcommand."""
    parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted optimization study",
        description="Resume a previously interrupted optimization study.",
    )

    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default="latest",
        help="Run ID to resume (default: latest)",
    )

    parser.add_argument(
        "--additional-trials",
        type=int,
        default=0,
        metavar="N",
        help="Run additional trials beyond original target",
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (useful for CI)",
    )

    parser.set_defaults(handler=execute, subcommand="resume")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the resume command."""
    storage = OptimizationStorage()
    output_format = getattr(args, "output_format", "text")
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

    # Check if run is resumable
    if run.completed_at is not None and args.additional_trials == 0:
        if output_format == "json":
            return CliResponse.success_response(
                command=COMMAND_NAME,
                data={
                    "run_id": run_id,
                    "status": "already_completed",
                    "total_trials": run.total_trials,
                },
                was_noop=True,
                warnings=["Run is already completed. Use --additional-trials N to run more trials"],
            )
        logger.info(f"Run {run_id} is already completed")
        logger.info("Use --additional-trials N to run more trials")
        return 0

    # Reconstruct configuration
    config_data = run_data.get("config", {})
    original_trials = config_data.get("number_of_trials", 100)
    completed_trials = run.total_trials

    remaining_trials = original_trials - completed_trials + args.additional_trials

    if remaining_trials <= 0:
        if output_format == "json":
            return CliResponse.success_response(
                command=COMMAND_NAME,
                data={
                    "run_id": run_id,
                    "status": "all_trials_completed",
                    "total_trials": original_trials,
                },
                was_noop=True,
                warnings=["All trials completed. Use --additional-trials N to run more trials"],
            )
        logger.info(f"Run {run_id} has completed all {original_trials} trials")
        if args.additional_trials == 0:
            logger.info("Use --additional-trials N to run more trials")
        return 0

    if not args.quiet and output_format == "text":
        print(f"Resuming optimization study: {run.study_name}")
        print(f"Run ID: {run_id}")
        print(f"Completed trials: {completed_trials}")
        print(f"Remaining trials: {remaining_trials}")
        print()

    # Attempt to reconstruct OptimizationConfig
    try:
        opt_config = _reconstruct_config(run_data, remaining_trials)
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to reconstruct configuration: {e}",
                details={"run_id": run_id},
            )
        logger.error(f"Failed to reconstruct configuration: {e}")
        logger.info("Try running a new study with the same configuration file")
        return 1

    # Create objective
    objective_name = config_data.get("objective_name", "sharpe")
    try:
        objective = create_objective_from_preset(objective_name)
    except ConfigValidationError as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to create objective: {e}",
            )
        logger.error(f"Failed to create objective: {e}")
        return 1

    # Create study manager with storage URL to load existing study
    study_manager = OptimizationStudyManager(opt_config)

    try:
        study = study_manager.create_or_load_study()
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.OPERATION_FAILED,
                message=f"Failed to load study: {e}",
            )
        logger.error(f"Failed to load study: {e}")
        return 1

    # Track results
    trial_results: list[TrialResult] = list(run.trials) if run.trials else []
    interrupted = False

    # Determine best from existing trials
    best_trial = None
    best_value = float("-inf") if opt_config.direction == "maximize" else float("inf")

    for trial_result in trial_results:
        is_better = (
            trial_result.objective_value > best_value
            if opt_config.direction == "maximize"
            else trial_result.objective_value < best_value
        )
        if is_better and trial_result.is_feasible:
            best_value = trial_result.objective_value
            best_trial = trial_result

    # Run remaining optimization trials
    try:
        for i in range(remaining_trials):
            trial_num = completed_trials + i
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            if not args.quiet and output_format == "text":
                print(
                    f"Trial {trial_num + 1}/{original_trials + args.additional_trials}...", end=" "
                )

            trial_value = _placeholder_evaluate(params, objective)
            study.tell(trial, trial_value)

            trial_result = TrialResult(
                trial_number=trial_num,
                parameters=params,
                objective_value=trial_value,
                is_feasible=trial_value > float("-inf"),
            )
            trial_results.append(trial_result)

            is_better = (
                trial_value > best_value
                if opt_config.direction == "maximize"
                else trial_value < best_value
            )
            if is_better:
                best_value = trial_value
                best_trial = trial_result

            if not args.quiet and output_format == "text":
                print(f"value={trial_value:.4f}")

    except KeyboardInterrupt:
        interrupted = True
        warnings.append("Optimization interrupted by user")
        if not args.quiet and output_format == "text":
            print("\nOptimization interrupted by user")

    completed_at = datetime.now()

    # Update and save results
    updated_run = OptimizationRun(
        run_id=run_id,
        study_name=run.study_name,
        started_at=run.started_at,
        completed_at=completed_at,
        config=opt_config,
        best_parameters=best_trial.parameters if best_trial else run.best_parameters,
        best_objective_value=best_value if best_trial else run.best_objective_value,
        total_trials=len(trial_results),
        feasible_trials=sum(1 for t in trial_results if t.is_feasible),
        trials=trial_results,
    )

    try:
        storage.save_run(updated_run)
    except Exception as e:
        warnings.append(f"Failed to save run: {e}")
        logger.warning(f"Failed to save run: {e}")

    # Output results
    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "run_id": run_id,
                "study_name": run.study_name,
                "total_trials": len(trial_results),
                "feasible_trials": sum(1 for t in trial_results if t.is_feasible),
                "best_objective_value": best_value if best_trial else run.best_objective_value,
                "best_parameters": best_trial.parameters if best_trial else run.best_parameters,
                "completed": not interrupted,
                "resumed_trials": remaining_trials if not interrupted else i + 1,
            },
            warnings=warnings,
        )

    updated_data = updated_run.to_dict()
    if not args.quiet:
        print()
    print(format_run_summary_text(updated_data))

    return 0


def _reconstruct_config(run_data: dict[str, Any], trials: int) -> OptimizationConfig:
    """
    Reconstruct OptimizationConfig from saved run data.

    This is a simplified reconstruction. Full reconstruction would require
    saving the complete ParameterSpace definition.
    """
    config_data = run_data.get("config", {})

    # Create minimal parameter space (in production, this would be loaded from storage)
    from gpt_trader.features.optimize.parameter_space.builder import ParameterSpaceBuilder

    builder = ParameterSpaceBuilder()
    builder.with_strategy_defaults()
    builder.with_risk_defaults()
    parameter_space = builder.build()

    return OptimizationConfig(
        study_name=config_data.get("study_name", run_data["study_name"]),
        parameter_space=parameter_space,
        objective_name=config_data.get("objective_name", "sharpe"),
        direction=config_data.get("direction", "maximize"),
        number_of_trials=trials,
        sampler_type=config_data.get("sampler_type", "tpe"),
        pruner_type=config_data.get("pruner_type", "median"),
    )


def _placeholder_evaluate(params: dict[str, Any], objective: Any) -> float:
    """Placeholder evaluation function (same as run.py)."""
    import hashlib
    import json

    param_str = json.dumps(params, sort_keys=True)
    hash_val = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16)
    return (hash_val % 1000) / 100.0 - 5.0
