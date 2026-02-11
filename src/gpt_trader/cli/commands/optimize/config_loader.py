"""YAML configuration loader for optimization CLI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from gpt_trader.cli.commands.optimize.registry import (
    DEFAULT_PARAMETER_GROUPS,
    add_parameter_groups,
    get_objective_spec,
    has_objective,
    has_parameter_group,
    list_objective_names,
    list_parameter_group_names,
)
from gpt_trader.features.optimize.parameter_space.builder import ParameterSpaceBuilder
from gpt_trader.features.optimize.types import OptimizationConfig, ParameterSpace


class ConfigValidationError(Exception):
    """Raised when configuration is invalid."""


@dataclass
class BacktestSettings:
    """Backtest configuration settings."""

    start_date: datetime
    end_date: datetime
    granularity: str = "FIVE_MINUTE"
    initial_equity: float = 100000.0


@dataclass
class StudySettings:
    """Study configuration settings."""

    name: str
    trials: int = 100
    sampler: str = "tpe"
    pruner: str | None = "median"
    seed: int | None = None
    timeout_seconds: int | None = None


@dataclass
class OptimizeCliConfig:
    """Complete CLI configuration for optimization."""

    study: StudySettings
    objective_name: str
    objective_kwargs: dict[str, Any] = field(default_factory=dict)
    strategy_type: str = "perps_baseline"
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD"])
    backtest: BacktestSettings | None = None
    parameter_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    include_parameter_groups: list[str] = field(default_factory=lambda: list(DEFAULT_PARAMETER_GROUPS))



def _normalize_parameter_groups(
    raw_value: Any | None,
) -> list[str]:
    if raw_value is None:
        candidate_groups: list[str] = list(DEFAULT_PARAMETER_GROUPS)
    elif isinstance(raw_value, str):
        candidate_groups = [raw_value]
    elif isinstance(raw_value, Sequence):
        candidate_groups = list(raw_value)
    else:
        raise ConfigValidationError("parameter_space.include_groups must be a list of strings")

    normalized: list[str] = []
    available = list_parameter_group_names()
    for group in candidate_groups:
        if not isinstance(group, str):
            raise ConfigValidationError("Parameter group names must be strings")
        if not has_parameter_group(group):
            raise ConfigValidationError(
                f"Unknown parameter group: {group}. Available: {', '.join(available)}"
            )
        if group not in normalized:
            normalized.append(group)

    return normalized or list(DEFAULT_PARAMETER_GROUPS)


def _validate_objective_name(raw_name: str | None) -> str:
    name = raw_name or "sharpe"
    if not has_objective(name):
        available = ", ".join(list_objective_names())
        raise ConfigValidationError(
            f"Unknown objective preset: {name}. Available: {available}"
        )
    return name

def load_config_file(config_path: Path) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigValidationError: If file cannot be loaded or parsed
    """
    if not config_path.exists():
        raise ConfigValidationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML in {config_path}: {e}") from e

    if not isinstance(config, dict):
        raise ConfigValidationError(f"Configuration must be a dictionary, got {type(config)}")

    return config


def parse_config(raw_config: dict[str, Any]) -> OptimizeCliConfig:
    """
    Parse raw configuration dictionary into typed config object.

    Args:
        raw_config: Dictionary from YAML file

    Returns:
        Typed configuration object

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Parse study settings
    study_raw = raw_config.get("study", {})
    if not study_raw.get("name"):
        raise ConfigValidationError("study.name is required")

    study = StudySettings(
        name=study_raw["name"],
        trials=study_raw.get("trials", 100),
        sampler=study_raw.get("sampler", "tpe"),
        pruner=study_raw.get("pruner", "median"),
        seed=study_raw.get("seed"),
        timeout_seconds=study_raw.get("timeout_seconds"),
    )

    # Parse objective
    objective_raw = raw_config.get("objective", {})
    objective_name = _validate_objective_name(objective_raw.get("preset"))
    objective_kwargs = {}

    if "constraints" in objective_raw:
        # Map constraint config to objective kwargs
        constraints = objective_raw["constraints"]
        if "max_drawdown_pct" in constraints:
            objective_kwargs["max_drawdown_pct"] = constraints["max_drawdown_pct"]
        if "max_var_95" in constraints:
            objective_kwargs["max_var_95"] = constraints["max_var_95"]
        if "max_var_99" in constraints:
            objective_kwargs["max_var_99"] = constraints["max_var_99"]
        if "min_trades" in constraints:
            objective_kwargs["min_trades"] = constraints["min_trades"]
        if "max_leverage" in constraints:
            objective_kwargs["max_leverage"] = constraints["max_leverage"]
        if "max_slippage_bps" in constraints:
            objective_kwargs["max_slippage_bps"] = constraints["max_slippage_bps"]
        if "min_fill_rate" in constraints:
            objective_kwargs["min_fill_rate"] = constraints["min_fill_rate"]
        if "max_consecutive_losses" in constraints:
            objective_kwargs["max_consecutive_losses"] = constraints["max_consecutive_losses"]
        if "min_win_rate" in constraints:
            objective_kwargs["min_win_rate"] = constraints["min_win_rate"]

    # Parse strategy
    strategy_raw = raw_config.get("strategy", {})
    strategy_type = strategy_raw.get("type", "perps_baseline")
    symbols = strategy_raw.get("symbols", ["BTC-USD"])

    # Parse backtest settings
    backtest = None
    backtest_raw = raw_config.get("backtest", {})
    if backtest_raw:
        start_date_str = backtest_raw.get("start_date")
        end_date_str = backtest_raw.get("end_date")

        if start_date_str and end_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
            except ValueError as e:
                raise ConfigValidationError(f"Invalid date format: {e}") from e

            backtest = BacktestSettings(
                start_date=start_date,
                end_date=end_date,
                granularity=backtest_raw.get("granularity", "FIVE_MINUTE"),
                initial_equity=backtest_raw.get("initial_equity", 100000.0),
            )

    # Parse parameter space
    param_space_raw = raw_config.get("parameter_space", {})
    include_groups = _normalize_parameter_groups(param_space_raw.get("include_groups"))
    parameter_overrides = param_space_raw.get("overrides", {})

    return OptimizeCliConfig(
        study=study,
        objective_name=objective_name,
        objective_kwargs=objective_kwargs,
        strategy_type=strategy_type,
        symbols=symbols if isinstance(symbols, list) else [symbols],
        backtest=backtest,
        parameter_overrides=parameter_overrides,
        include_parameter_groups=include_groups,
    )


def merge_cli_overrides(
    config: OptimizeCliConfig,
    cli_args: dict[str, Any],
) -> OptimizeCliConfig:
    """
    Merge CLI argument overrides into configuration.

    CLI arguments take precedence over config file values.

    Args:
        config: Base configuration from file
        cli_args: Dictionary of CLI arguments

    Returns:
        Updated configuration
    """
    # Override study settings
    if cli_args.get("trials"):
        config.study.trials = cli_args["trials"]
    if cli_args.get("sampler"):
        config.study.sampler = cli_args["sampler"]
    if cli_args.get("seed"):
        config.study.seed = cli_args["seed"]
    if cli_args.get("timeout"):
        config.study.timeout_seconds = cli_args["timeout"]

    # Override objective
    if cli_args.get("objective"):
        config.objective_name = cli_args["objective"]

    # Override strategy settings
    if cli_args.get("strategy"):
        config.strategy_type = cli_args["strategy"]
    if cli_args.get("symbols"):
        config.symbols = cli_args["symbols"]

    # Override backtest dates
    if cli_args.get("start_date") and cli_args.get("end_date"):
        if config.backtest is None:
            config.backtest = BacktestSettings(
                start_date=cli_args["start_date"],
                end_date=cli_args["end_date"],
            )
        else:
            config.backtest.start_date = cli_args["start_date"]
            config.backtest.end_date = cli_args["end_date"]

    if cli_args.get("granularity") and config.backtest:
        config.backtest.granularity = cli_args["granularity"]

    return config


def create_objective_from_preset(preset_name: str, **kwargs: Any) -> Any:
    """
    Create an objective function from a preset name.

    Args:
        preset_name: Name of the objective preset
        **kwargs: Additional arguments for the objective factory

    Returns:
        Objective function instance

    Raises:
        ConfigValidationError: If preset is unknown
    """
    validated_name = _validate_objective_name(preset_name)
    spec = get_objective_spec(validated_name)
    return spec.factory(**kwargs)


def get_objective_direction(preset_name: str) -> str:
    """
    Get the optimization direction for an objective preset.

    Args:
        preset_name: Name of the objective preset

    Returns:
        "maximize" or "minimize"

    Raises:
        ConfigValidationError: If preset is unknown
    """
    validated_name = _validate_objective_name(preset_name)
    spec = get_objective_spec(validated_name)
    return spec.direction


def build_parameter_space_from_config(config: OptimizeCliConfig) -> ParameterSpace:
    """
    Build a parameter space from configuration.

    Args:
        config: CLI configuration

    Returns:
        Configured ParameterSpace
    """
    builder = ParameterSpaceBuilder()
    builder = add_parameter_groups(builder, config.include_parameter_groups)

    # Apply overrides
    for param_name, override_config in config.parameter_overrides.items():
        builder.override(
            param_name,
            low=override_config.get("low"),
            high=override_config.get("high"),
            choices=override_config.get("choices"),
        )

    return builder.build()


def build_optimization_config(cli_config: OptimizeCliConfig) -> OptimizationConfig:
    """
    Build an OptimizationConfig from CLI configuration.

    Args:
        cli_config: CLI configuration object

    Returns:
        OptimizationConfig for the study manager
    """
    parameter_space = build_parameter_space_from_config(cli_config)

    return OptimizationConfig(
        study_name=cli_config.study.name,
        parameter_space=parameter_space,
        objective_name=cli_config.objective_name,
        direction=get_objective_direction(cli_config.objective_name),
        number_of_trials=cli_config.study.trials,
        timeout_seconds=cli_config.study.timeout_seconds,
        sampler_type=cli_config.study.sampler,
        pruner_type=cli_config.study.pruner,
        seed=cli_config.study.seed,
    )


def create_default_config(
    study_name: str,
    objective: str = "sharpe",
    trials: int = 100,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> OptimizeCliConfig:
    """
    Create a default configuration with minimal settings.

    Args:
        study_name: Name for the optimization study
        objective: Objective preset name
        trials: Number of trials
        symbols: Trading symbols
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Default configuration object
    """
    study = StudySettings(name=study_name, trials=trials)

    backtest = None
    if start_date and end_date:
        backtest = BacktestSettings(start_date=start_date, end_date=end_date)

    return OptimizeCliConfig(
        study=study,
        objective_name=objective,
        symbols=symbols or ["BTC-USD"],
        backtest=backtest,
    )
