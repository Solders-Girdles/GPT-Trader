"""Run command for optimization CLI."""

from __future__ import annotations

import asyncio
import uuid
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_trader.cli.commands.optimize.config_loader import (
    OBJECTIVE_PRESETS,
    ConfigValidationError,
    build_optimization_config,
    build_parameter_space_from_config,
    create_default_config,
    create_objective_from_preset,
    get_objective_direction,
    load_config_file,
    merge_cli_overrides,
    parse_config,
)
from gpt_trader.cli.commands.optimize.formatters import format_run_summary_text
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.optimize.persistence.storage import (
    OptimizationRun,
    OptimizationStorage,
)
from gpt_trader.features.optimize.runner.batch_runner import BatchBacktestRunner, TrialResult
from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, enable_console=True)

COMMAND_NAME = "optimize run"


def register(subparsers: Any) -> None:
    """Register the run subcommand."""
    parser = subparsers.add_parser(
        "run",
        help="Run an optimization study",
        description="Run a strategy optimization study using Optuna.",
    )

    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="YAML configuration file for the study",
    )

    # Objective preset
    parser.add_argument(
        "--objective",
        type=str,
        choices=list(OBJECTIVE_PRESETS.keys()),
        default="sharpe",
        help="Objective function preset (default: sharpe)",
    )

    # Strategy settings
    parser.add_argument(
        "--strategy",
        type=str,
        default="perps_baseline",
        help="Strategy type to optimize (default: perps_baseline)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC-USD"],
        help="Trading symbols (default: BTC-USD)",
    )

    # Study settings
    parser.add_argument(
        "--name",
        type=str,
        help="Study name (default: auto-generated)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Optuna sampler type (default: tpe)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum optimization time in seconds",
    )

    # Backtest settings
    parser.add_argument(
        "--start-date",
        type=str,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="FIVE_MINUTE",
        help="Candle granularity (default: FIVE_MINUTE)",
    )

    # Output settings
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

    # Development options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running optimization",
    )

    parser.set_defaults(handler=execute, subcommand="run")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the run command."""
    output_format = getattr(args, "output_format", "text")

    try:
        config = _build_config_from_args(args)
    except ConfigValidationError as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Configuration error: {e}",
            )
        logger.error(f"Configuration error: {e}")
        return 1

    if args.dry_run:
        return _handle_dry_run(config, args, output_format)

    return _run_optimization(config, args, output_format)


def _build_config_from_args(args: Namespace):
    """Build configuration from command arguments."""
    if args.config:
        # Load from YAML file
        raw_config = load_config_file(args.config)
        config = parse_config(raw_config)

        # Merge CLI overrides
        cli_overrides = _extract_cli_overrides(args)
        config = merge_cli_overrides(config, cli_overrides)
    else:
        # Build from CLI arguments
        study_name = args.name or f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start_date = None
        end_date = None
        if args.start_date and args.end_date:
            start_date = datetime.fromisoformat(args.start_date)
            end_date = datetime.fromisoformat(args.end_date)

        config = create_default_config(
            study_name=study_name,
            objective=args.objective,
            trials=args.trials,
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
        )

        config.strategy_type = args.strategy
        config.study.sampler = args.sampler
        config.study.seed = args.seed
        config.study.timeout_seconds = args.timeout

        if config.backtest:
            config.backtest.granularity = args.granularity

    return config


def _extract_cli_overrides(args: Namespace) -> dict[str, Any]:
    """Extract CLI arguments as override dictionary."""
    overrides = {}

    if args.trials:
        overrides["trials"] = args.trials
    if args.sampler:
        overrides["sampler"] = args.sampler
    if args.seed:
        overrides["seed"] = args.seed
    if args.timeout:
        overrides["timeout"] = args.timeout
    if args.objective:
        overrides["objective"] = args.objective
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.symbols:
        overrides["symbols"] = args.symbols
    if args.start_date:
        overrides["start_date"] = datetime.fromisoformat(args.start_date)
    if args.end_date:
        overrides["end_date"] = datetime.fromisoformat(args.end_date)
    if args.granularity:
        overrides["granularity"] = args.granularity

    return overrides


def _handle_dry_run(config, args: Namespace, output_format: str) -> CliResponse | int:
    """Handle dry run mode - validate and show configuration."""
    # Build parameter space info
    param_info = []
    try:
        param_space = build_parameter_space_from_config(config)
        for param in param_space.all_parameters:
            info = {"name": param.name}
            if param.low is not None and param.high is not None:
                info["range"] = [param.low, param.high]
            elif param.choices:
                info["choices"] = param.choices
            param_info.append(info)
    except Exception as e:
        param_info = None
        logger.warning(f"Could not build parameter space: {e}")

    if output_format == "json":
        data = {
            "study_name": config.study.name,
            "objective": config.objective_name,
            "strategy": config.strategy_type,
            "symbols": config.symbols,
            "trials": config.study.trials,
            "sampler": config.study.sampler,
            "parameter_count": len(param_info) if param_info else 0,
            "parameters": param_info[:10] if param_info else [],  # Limit for readability
        }
        if config.backtest:
            data["backtest"] = {
                "start_date": config.backtest.start_date.isoformat() if config.backtest.start_date else None,
                "end_date": config.backtest.end_date.isoformat() if config.backtest.end_date else None,
                "granularity": config.backtest.granularity,
            }
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=data,
            was_noop=True,
        )

    # Text format
    print("Dry run - configuration validated successfully")
    print()
    print(f"Study Name: {config.study.name}")
    print(f"Objective: {config.objective_name}")
    print(f"Strategy: {config.strategy_type}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Trials: {config.study.trials}")
    print(f"Sampler: {config.study.sampler}")

    if config.backtest:
        print(f"Start Date: {config.backtest.start_date.date()}")
        print(f"End Date: {config.backtest.end_date.date()}")
        print(f"Granularity: {config.backtest.granularity}")

    if param_info:
        print(f"\nParameter Space: {len(param_info)} parameters")
        for info in param_info[:5]:
            if "range" in info:
                print(f"  {info['name']}: [{info['range'][0]}, {info['range'][1]}]")
            elif "choices" in info:
                print(f"  {info['name']}: {info['choices']}")
        if len(param_info) > 5:
            print(f"  ... and {len(param_info) - 5} more")

    return 0


def _run_optimization(config, args: Namespace, output_format: str) -> CliResponse | int:
    """Run the actual optimization study."""
    run_id = f"opt_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    storage = OptimizationStorage()
    warnings: list[str] = []

    if not args.quiet and output_format == "text":
        print(f"Starting optimization study: {config.study.name}")
        print(f"Run ID: {run_id}")
        print(f"Objective: {config.objective_name}")
        print(f"Trials: {config.study.trials}")
        print()

    # Create optimization config
    try:
        opt_config = build_optimization_config(config)
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to build optimization config: {e}",
            )
        logger.error(f"Failed to build optimization config: {e}")
        return 1

    # Create objective
    try:
        objective = create_objective_from_preset(
            config.objective_name,
            **config.objective_kwargs,
        )
    except ConfigValidationError as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to create objective: {e}",
            )
        logger.error(f"Failed to create objective: {e}")
        return 1

    # Create study manager
    study_manager = OptimizationStudyManager(opt_config)

    try:
        study = study_manager.create_or_load_study()
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.OPERATION_FAILED,
                message=f"Failed to create study: {e}",
            )
        logger.error(f"Failed to create study: {e}")
        return 1

    # Track results
    started_at = datetime.now()
    trial_results: list[TrialResult] = []
    best_trial = None
    best_value = float("-inf") if opt_config.direction == "maximize" else float("inf")
    interrupted = False

    # Run optimization loop
    try:
        for trial_num in range(opt_config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            if not args.quiet and output_format == "text":
                print(f"Trial {trial_num + 1}/{opt_config.number_of_trials}...", end=" ")

            trial_value = _placeholder_evaluate(params, objective)
            study.tell(trial, trial_value)

            # Track results
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

    # Save results
    optimization_run = OptimizationRun(
        run_id=run_id,
        study_name=config.study.name,
        started_at=started_at,
        completed_at=completed_at,
        config=opt_config,
        best_parameters=best_trial.parameters if best_trial else None,
        best_objective_value=best_value if best_trial else None,
        total_trials=len(trial_results),
        feasible_trials=sum(1 for t in trial_results if t.is_feasible),
        trials=trial_results,
    )

    try:
        storage.save_run(optimization_run)
    except Exception as e:
        warnings.append(f"Failed to save run: {e}")
        logger.warning(f"Failed to save run: {e}")

    # Build result data
    run_data = optimization_run.to_dict()

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "run_id": run_id,
                "study_name": config.study.name,
                "objective": config.objective_name,
                "total_trials": len(trial_results),
                "feasible_trials": sum(1 for t in trial_results if t.is_feasible),
                "best_objective_value": best_value if best_trial else None,
                "best_parameters": best_trial.parameters if best_trial else None,
                "completed": not interrupted,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
            warnings=warnings,
        )

    # Text format
    if not args.quiet:
        print()
    print(format_run_summary_text(run_data))

    return 0


def _placeholder_evaluate(params: dict[str, Any], objective) -> float:
    """
    Placeholder evaluation function.

    In a real implementation, this would:
    1. Configure a strategy with the given parameters
    2. Run a backtest using BatchBacktestRunner
    3. Calculate metrics and evaluate the objective

    For now, returns a random-ish value based on params.
    """
    # Simple placeholder that varies with parameters
    # This should be replaced with actual backtest execution
    import hashlib
    import json

    param_str = json.dumps(params, sort_keys=True)
    hash_val = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16)
    return (hash_val % 1000) / 100.0 - 5.0  # Range roughly [-5, 5]
