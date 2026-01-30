"""Run command for optimization CLI."""

from __future__ import annotations

import asyncio
import uuid
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from gpt_trader.backtesting.data.manager import (
    HistoricalDataManager,
    create_coinbase_data_provider,
    create_offline_data_provider,
)
from gpt_trader.cli.commands.optimize.config_loader import (
    OBJECTIVE_PRESETS,
    ConfigValidationError,
    OptimizeCliConfig,
    build_optimization_config,
    build_parameter_space_from_config,
    create_default_config,
    create_objective_from_preset,
    load_config_file,
    merge_cli_overrides,
    parse_config,
)
from gpt_trader.cli.commands.optimize.formatters import format_run_summary_text
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.config.strategy_types import normalize_strategy_type, resolve_strategy_type
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.live_trade.strategies.base import StrategyProtocol
from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    PerpsStrategy,
    PerpsStrategyConfig,
    SpotStrategy,
    SpotStrategyConfig,
)
from gpt_trader.features.optimize.persistence.storage import (
    OptimizationRun,
    OptimizationStorage,
)
from gpt_trader.features.optimize.runner.batch_runner import (
    BatchBacktestRunner,
    TrialResult,
)
from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.research.artifacts import StrategyArtifact, StrategyArtifactStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

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
        default=None,
        help="Objective function preset (default: sharpe when no --config)",
    )

    # Strategy settings
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy type to optimize (default: baseline when no --config; aliases: perps_baseline, spot)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Trading symbols (default: BTC-USD when no --config)",
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
        default=None,
        help="Number of optimization trials (default: 100 when no --config)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "cmaes", "random"],
        default=None,
        help="Optuna sampler type (default: tpe when no --config)",
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
        default=None,
        help="Candle granularity (default: FIVE_MINUTE when no --config)",
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


def _build_config_from_args(args: Namespace) -> OptimizeCliConfig:
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
        objective = args.objective if args.objective is not None else "sharpe"
        trials = args.trials if args.trials is not None else 100
        symbols = args.symbols if args.symbols is not None else ["BTC-USD"]
        strategy_raw = args.strategy if args.strategy is not None else "baseline"
        strategy_type, strategy_variant = resolve_strategy_type(strategy_raw)
        sampler = args.sampler if args.sampler is not None else "tpe"
        granularity = args.granularity if args.granularity is not None else "FIVE_MINUTE"

        start_date = None
        end_date = None
        if args.start_date and args.end_date:
            start_date = datetime.fromisoformat(args.start_date)
            end_date = datetime.fromisoformat(args.end_date)

        config = create_default_config(
            study_name=study_name,
            objective=objective,
            trials=trials,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        config.strategy_type = strategy_type
        config.strategy_variant = strategy_variant
        config.study.sampler = sampler
        config.study.seed = args.seed
        config.study.timeout_seconds = args.timeout

        if config.backtest:
            config.backtest.granularity = granularity

    return config


def _extract_cli_overrides(args: Namespace) -> dict[str, Any]:
    """Extract CLI arguments as override dictionary."""
    overrides = {}

    if args.trials is not None:
        overrides["trials"] = args.trials
    if args.sampler is not None:
        overrides["sampler"] = args.sampler
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.timeout is not None:
        overrides["timeout"] = args.timeout
    if args.objective is not None:
        overrides["objective"] = args.objective
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.symbols:
        overrides["symbols"] = args.symbols
    if args.start_date is not None:
        overrides["start_date"] = datetime.fromisoformat(args.start_date)
    if args.end_date is not None:
        overrides["end_date"] = datetime.fromisoformat(args.end_date)
    if args.granularity is not None:
        overrides["granularity"] = args.granularity

    return overrides


def _handle_dry_run(
    config: OptimizeCliConfig, args: Namespace, output_format: str
) -> CliResponse | int:
    """Handle dry run mode - validate and show configuration."""
    # Build parameter space info
    param_info: list[dict[str, Any]] | None = []
    try:
        param_space = build_parameter_space_from_config(config)
        for param in param_space.all_parameters:
            info: dict[str, Any] = {"name": param.name}
            if param.low is not None and param.high is not None:
                info["range"] = [param.low, param.high]
            elif param.choices:
                info["choices"] = param.choices
            if param_info is not None:
                param_info.append(info)
    except Exception as e:
        param_info = None
        logger.warning(f"Could not build parameter space: {e}")

    if output_format == "json":
        strategy_label = config.strategy_type
        if config.strategy_variant:
            strategy_label = f"{config.strategy_type} ({config.strategy_variant})"
        data = {
            "study_name": config.study.name,
            "objective": config.objective_name,
            "strategy": strategy_label,
            "strategy_type": config.strategy_type,
            "strategy_variant": config.strategy_variant,
            "symbols": config.symbols,
            "trials": config.study.trials,
            "sampler": config.study.sampler,
            "parameter_count": len(param_info) if param_info else 0,
            "parameters": param_info[:10] if param_info else [],  # Limit for readability
        }
        if config.backtest:
            data["backtest"] = {
                "start_date": (
                    config.backtest.start_date.isoformat() if config.backtest.start_date else None
                ),
                "end_date": (
                    config.backtest.end_date.isoformat() if config.backtest.end_date else None
                ),
                "granularity": config.backtest.granularity,
            }
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data=data,
            was_noop=True,
        )

    # Text format
    strategy_label = config.strategy_type
    if config.strategy_variant:
        strategy_label = f"{config.strategy_type} ({config.strategy_variant})"
    print("Dry run - configuration validated successfully")
    print()
    print(f"Study Name: {config.study.name}")
    print(f"Objective: {config.objective_name}")
    print(f"Strategy: {strategy_label}")
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


def _create_data_provider() -> HistoricalDataManager:
    """Create data provider using environment variables for credentials."""
    import json
    import os

    data_source = os.environ.get("BACKTEST_DATA_SOURCE", "coinbase").lower()
    if data_source in {"offline", "cache", "dataset"}:
        cache_dir = os.environ.get("BACKTEST_DATA_DIR")
        return create_offline_data_provider(cache_dir=cache_dir, validate_quality=True)

    # Reuse existing credential loading logic from container.py
    api_key_name = None
    private_key = None

    creds_file = os.environ.get("COINBASE_CREDENTIALS_FILE")
    if creds_file:
        path = Path(creds_file)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                api_key_name = data.get("name")
                private_key = data.get("privateKey")

    if not api_key_name:
        api_key_name = os.environ.get("COINBASE_API_KEY_NAME")
    if not private_key:
        private_key = os.environ.get("COINBASE_PRIVATE_KEY")

    if not api_key_name or not private_key:
        raise ValueError(
            "Coinbase credentials required for optimization. "
            "Set COINBASE_CREDENTIALS_FILE or COINBASE_API_KEY_NAME + COINBASE_PRIVATE_KEY"
        )

    auth = SimpleAuth(key_name=api_key_name, private_key=private_key)
    client = CoinbaseClient(auth=auth)

    return create_coinbase_data_provider(client=client, validate_quality=True)


def _create_strategy_factory(
    strategy_type: str,
    strategy_variant: str | None,
) -> Callable[[dict[str, Any]], StrategyProtocol]:
    """Create a strategy factory for the given strategy type."""
    resolved_type, resolved_variant = resolve_strategy_type(strategy_type, variant=strategy_variant)

    def factory(params: dict[str, Any]) -> StrategyProtocol:
        if resolved_type == "baseline" and resolved_variant == "spot":
            # Filter params for SpotStrategyConfig?
            # For now, we assume params are valid kwargs, but in reality Optuna
            # might pass extra stuff if we aren't careful.
            # Ideally we'd filter against the dataclass fields.
            # But BaseStrategyConfig allows extra kwargs usually via ** if set up right,
            # or we rely on the parameter space builder to only provide valid keys.
            # For now, pass all params.
            try:
                spot_config = SpotStrategyConfig(**params)
                return SpotStrategy(config=spot_config)
            except TypeError:
                # If strict typing fails, try filtering or fallback
                # Simplified for now as per plan
                spot_config = SpotStrategyConfig(**params)
                return SpotStrategy(config=spot_config)
        if resolved_type != "baseline":
            logger.warning(
                "Optimization currently supports baseline strategies only; "
                "falling back to baseline perps",
                strategy_type=resolved_type,
            )
        # baseline perps or default
        perps_config = PerpsStrategyConfig(**params)
        return PerpsStrategy(config=perps_config)

    return factory


def _run_optimization(
    config: OptimizeCliConfig, args: Namespace, output_format: str
) -> CliResponse | int:
    """Run the actual optimization study."""
    # Validate backtest settings are present
    if not config.backtest or not config.backtest.start_date or not config.backtest.end_date:
        message = (
            "Optimization requires a backtest period. "
            "Please specify --start-date and --end-date, or provide them in the configuration file."
        )
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=message,
            )
        logger.error(message)
        return 1

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

    # Initialize Data and Strategy Factory
    try:
        data_provider = _create_data_provider()
        strategy_factory = _create_strategy_factory(config.strategy_type, config.strategy_variant)

        # Initialize Batch Runner
        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=config.symbols,
            granularity=config.backtest.granularity,
            start_date=config.backtest.start_date,
            end_date=config.backtest.end_date,
            strategy_factory=strategy_factory,
            objective=objective,
        )
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.OPERATION_FAILED,
                message=f"Failed to initialize runner: {e}",
            )
        logger.error(f"Failed to initialize runner: {e}")
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
    async def run_optimization_loop() -> None:
        nonlocal best_value, best_trial, interrupted
        try:
            for trial_num in range(opt_config.number_of_trials):
                trial = study.ask()
                params = study_manager.suggest_parameters(trial)

                if not args.quiet and output_format == "text":
                    print(
                        f"Trial {trial_num + 1}/{opt_config.number_of_trials}...",
                        end=" ",
                        flush=True,
                    )

                # Run the actual trial
                trial_result = await runner.run_trial(trial.number, params)

                # Report to Optuna
                study.tell(trial, trial_result.objective_value)

                # Track results
                trial_results.append(trial_result)

                if trial_result.is_feasible:
                    is_better = (
                        trial_result.objective_value > best_value
                        if opt_config.direction == "maximize"
                        else trial_result.objective_value < best_value
                    )
                    if is_better:
                        best_value = trial_result.objective_value
                        best_trial = trial_result

                if not args.quiet and output_format == "text":
                    status = "OK" if trial_result.is_feasible else "FAIL"
                    print(f"{status} value={trial_result.objective_value:.4f}")

        except KeyboardInterrupt:
            interrupted = True
            warnings.append("Optimization interrupted by user")
            if not args.quiet and output_format == "text":
                print("\nOptimization interrupted by user")

    # Execute loop
    try:
        asyncio.run(run_optimization_loop())
    except Exception as e:
        logger.error(f"Optimization loop failed: {e}")
        warnings.append(f"Optimization failed: {e}")

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

    results_path: Path | None = None
    try:
        results_path = storage.save_run(optimization_run)
    except Exception as e:
        warnings.append(f"Failed to save run: {e}")
        logger.warning(f"Failed to save run: {e}")

    artifact_emission: tuple[StrategyArtifact, Path] | None = None
    if best_trial is None:
        warnings.append("No feasible trials; strategy artifact not created.")
    else:
        try:
            artifact_emission = _emit_strategy_artifact(
                best_trial=best_trial,
                optimization_config=opt_config,
                cli_config=config,
                run_id=run_id,
                results_path=results_path,
                total_trials=len(trial_results),
                feasible_trials=sum(1 for t in trial_results if t.is_feasible),
                config_path=args.config if hasattr(args, "config") else None,
            )
        except Exception as e:
            warnings.append(f"Failed to create strategy artifact: {e}")
            logger.warning("Failed to create strategy artifact", error=str(e))

    # Build result data
    run_data = optimization_run.to_dict()

    if output_format == "json":
        artifact_payload = None
        if artifact_emission is not None:
            artifact, artifact_path = artifact_emission
            artifact_payload = {
                "artifact_id": artifact.artifact_id,
                "path": str(artifact_path),
            }
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
                "strategy_artifact": artifact_payload,
            },
            warnings=warnings,
        )

    # Text format
    if not args.quiet:
        print()
    print(format_run_summary_text(run_data))
    if artifact_emission is not None and not args.quiet:
        artifact, artifact_path = artifact_emission
        print(f"Strategy artifact: {artifact.artifact_id} ({artifact_path})")

    return 0


def _emit_strategy_artifact(
    *,
    best_trial: TrialResult,
    optimization_config: Any,
    cli_config: OptimizeCliConfig,
    run_id: str,
    results_path: Path | None,
    total_trials: int,
    feasible_trials: int,
    config_path: Path | None,
) -> tuple[StrategyArtifact, Path]:
    strategy_type, strategy_variant = resolve_strategy_type(
        cli_config.strategy_type, variant=cli_config.strategy_variant
    )
    artifact_strategy_type = normalize_strategy_type(strategy_type)
    interval = _granularity_seconds(
        cli_config.backtest.granularity if cli_config.backtest else None
    )

    (
        strategy_parameters,
        risk_parameters,
        simulation_parameters,
        extra_parameters,
    ) = _split_best_parameters(best_trial.parameters, optimization_config.parameter_space)

    risk_parameters = _normalize_risk_parameters(risk_parameters, strategy_variant)
    strategy_parameters = _merge_risk_into_strategy(
        strategy_parameters, risk_parameters, strategy_type, strategy_variant
    )

    strategy_parameters = _make_json_ready(strategy_parameters)
    risk_parameters = _make_json_ready(risk_parameters)
    simulation_parameters = _make_json_ready(simulation_parameters)
    extra_parameters = _make_json_ready(extra_parameters)

    metrics: dict[str, Any] = {
        "objective_name": cli_config.objective_name,
        "objective_value": best_trial.objective_value,
    }
    if best_trial.backtest_result is not None:
        metrics["backtest_result"] = _make_json_ready(best_trial.backtest_result)
    if best_trial.risk_metrics is not None:
        metrics["risk_metrics"] = _make_json_ready(best_trial.risk_metrics)
    if best_trial.trade_statistics is not None:
        metrics["trade_statistics"] = _make_json_ready(best_trial.trade_statistics)

    validation: dict[str, Any] = {
        "feasible": best_trial.is_feasible,
        "study_name": cli_config.study.name,
        "run_id": run_id,
        "objective_name": cli_config.objective_name,
        "direction": optimization_config.direction,
        "total_trials": total_trials,
        "feasible_trials": feasible_trials,
        "strategy_variant": strategy_variant,
    }

    if cli_config.backtest:
        validation["backtest"] = {
            "start_date": cli_config.backtest.start_date.isoformat(),
            "end_date": cli_config.backtest.end_date.isoformat(),
            "granularity": cli_config.backtest.granularity,
            "initial_equity": cli_config.backtest.initial_equity,
        }

    if simulation_parameters:
        validation["simulation_parameters"] = simulation_parameters
    if extra_parameters:
        validation["unmapped_parameters"] = extra_parameters

    evidence_paths: list[str] = []
    if results_path is not None:
        evidence_paths.append(str(results_path))
    if config_path is not None:
        evidence_paths.append(str(config_path))

    artifact = StrategyArtifact.create(
        strategy_type=artifact_strategy_type,
        symbols=list(cli_config.symbols),
        interval=interval,
        strategy_parameters=strategy_parameters,
        risk_parameters=risk_parameters,
        metrics=metrics,
        validation=validation,
        evidence_paths=evidence_paths,
        source=f"optimize:{run_id}",
    )

    store = StrategyArtifactStore()
    artifact_path = store.save(artifact)
    return artifact, artifact_path


def _split_best_parameters(
    parameters: dict[str, Any],
    parameter_space: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    strategy_names = {param.name for param in parameter_space.strategy_parameters}
    risk_names = {param.name for param in parameter_space.risk_parameters}
    simulation_names = {param.name for param in parameter_space.simulation_parameters}
    strategy_parameters = {
        name: value for name, value in parameters.items() if name in strategy_names
    }
    risk_parameters = {name: value for name, value in parameters.items() if name in risk_names}
    simulation_parameters = {
        name: value for name, value in parameters.items() if name in simulation_names
    }
    known_names = strategy_names | risk_names | simulation_names
    extra_parameters = {
        name: value for name, value in parameters.items() if name not in known_names
    }
    return strategy_parameters, risk_parameters, simulation_parameters, extra_parameters


def _normalize_risk_parameters(
    risk_parameters: dict[str, Any], strategy_variant: str | None
) -> dict[str, Any]:
    updated = dict(risk_parameters)
    if strategy_variant == "spot":
        updated["target_leverage"] = 1
        updated["max_leverage"] = 1
    return updated


def _merge_risk_into_strategy(
    strategy_parameters: dict[str, Any],
    risk_parameters: dict[str, Any],
    strategy_type: str,
    strategy_variant: str | None,
) -> dict[str, Any]:
    merged = dict(strategy_parameters)
    if normalize_strategy_type(strategy_type) == "baseline":
        for key in ("position_fraction", "target_leverage", "max_leverage"):
            if key in risk_parameters and key not in merged:
                merged[key] = risk_parameters[key]
        if strategy_variant == "spot":
            merged.setdefault("enable_shorts", False)
            merged.setdefault("target_leverage", 1)
            merged.setdefault("max_leverage", 1)
        elif strategy_variant in ("perps", None):
            merged.setdefault("enable_shorts", True)
    return merged


def _granularity_seconds(granularity: str | None) -> int | None:
    if granularity is None:
        return None
    mapping = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "TWO_HOUR": 7200,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }
    seconds = mapping.get(granularity)
    if seconds is None:
        logger.warning("Unknown granularity for artifact interval", granularity=granularity)
    return seconds


def _make_json_ready(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _make_json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _make_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_ready(item) for item in value]
    return value
