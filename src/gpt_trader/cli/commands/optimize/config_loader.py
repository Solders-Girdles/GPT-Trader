"""YAML configuration loader for optimization CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


_PARAMETER_SPACE_MISSING = object()


@dataclass(frozen=True)
class OptimizePresetSelection:
    """Selected optimize preset names for profile/scenario overrides."""

    profile: str | None
    scenario: str | None


@dataclass
class _PresetMergeReport:
    merged: dict[str, Any]
    missing_keys: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


_PRESET_CONTAINER_KEYS = ("presets", "base", "profiles", "scenarios")
_EXTENDABLE_OVERRIDE_PATHS: tuple[tuple[str, ...], ...] = (
    ("parameter_space", "overrides"),
    ("objective", "constraints"),
)


def resolve_optimize_preset_inheritance(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve base/profile/scenario optimize presets into a single config."""
    preset_container = _find_preset_container(raw_config)
    if preset_container is None:
        return raw_config

    presets = preset_container.get("presets", preset_container)
    if not isinstance(presets, Mapping):
        raise ConfigValidationError("optimize presets must be a mapping")

    base = _require_mapping(presets.get("base"), "presets.base")
    profiles = _ensure_mapping(presets.get("profiles"), "presets.profiles")
    scenarios = _ensure_mapping(presets.get("scenarios"), "presets.scenarios")

    # Do not inherit top-level profile/scenario selectors when presets are
    # defined under optimize.*; selectors must come from the same container.
    profile_raw = preset_container.get("profile")
    scenario_raw = preset_container.get("scenario")
    if preset_container is raw_config:
        profile_raw = raw_config.get("profile")
        scenario_raw = raw_config.get("scenario")

    selection = OptimizePresetSelection(
        profile=_normalize_override_name(profile_raw, "profile"),
        scenario=_normalize_override_name(scenario_raw, "scenario"),
    )

    profile_override = _resolve_named_override(
        profiles,
        selection.profile,
        label="profile preset",
    )
    scenario_override = _resolve_named_override(
        scenarios,
        selection.scenario,
        label="scenario preset",
    )

    merged = _merge_preset_layer(
        base,
        _normalize_override_entry(profile_override, label=selection.profile, kind="profile"),
        source="profile",
        selection_name=selection.profile,
    )
    merged = _merge_preset_layer(
        merged,
        _normalize_override_entry(scenario_override, label=selection.scenario, kind="scenario"),
        source="scenario",
        selection_name=selection.scenario,
    )

    inline_overrides = _extract_inline_overrides(preset_container)
    if inline_overrides:
        merged = _merge_preset_layer(
            merged,
            inline_overrides,
            source="inline",
            selection_name=None,
        )

    return merged


def _find_preset_container(raw_config: dict[str, Any]) -> Mapping[str, Any] | None:
    if "optimize" in raw_config and isinstance(raw_config["optimize"], Mapping):
        optimize_section = raw_config["optimize"]
        if _has_preset_keys(optimize_section):
            return optimize_section
    if _has_preset_keys(raw_config):
        return raw_config
    return None


def _has_preset_keys(candidate: Mapping[str, Any]) -> bool:
    if "presets" in candidate:
        return True
    return all(key in candidate for key in ("base", "profiles", "scenarios"))


def _normalize_override_name(raw_value: Any, label: str) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ConfigValidationError(f"{label} selection must be a string")
    normalized = raw_value.strip()
    return normalized or None


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{label} must be a mapping")
    return dict(value)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        raise ConfigValidationError(f"{label} is required")
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{label} must be a mapping")
    return dict(value)


def _resolve_named_override(
    overrides: Mapping[str, Any],
    name: str | None,
    *,
    label: str,
) -> Any:
    if name is None:
        return {}
    if name not in overrides:
        available = ", ".join(sorted(overrides.keys())) or "none"
        raise ConfigValidationError(f"Unknown {label}: {name}. Available: {available}")
    return overrides[name]


def _normalize_override_entry(
    entry: Any,
    *,
    label: str | None,
    kind: str,
) -> dict[str, Any]:
    if entry is None:
        return {}
    if not isinstance(entry, Mapping):
        name = label or "unknown"
        raise ConfigValidationError(f"{kind} preset '{name}' overrides must be a mapping")
    if not entry:
        return {}
    if "overrides" in entry:
        overrides = entry.get("overrides")
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            name = label or "unknown"
            raise ConfigValidationError(f"{kind} preset '{name}' overrides must be a mapping")
        return dict(overrides)
    return dict(entry)


def _extract_inline_overrides(container: Mapping[str, Any]) -> dict[str, Any]:
    inline: dict[str, Any] = {}
    for key, value in container.items():
        if key in {"presets", "base", "profiles", "scenarios", "profile", "scenario"}:
            continue
        inline[key] = value
    return inline


def _merge_preset_layer(
    base: dict[str, Any],
    overrides: dict[str, Any],
    *,
    source: str,
    selection_name: str | None,
) -> dict[str, Any]:
    if not overrides:
        return dict(base)

    report = _merge_preset_values(base, overrides, path=())
    if report.missing_keys or report.conflicts:
        label = _format_override_label(source, selection_name)
        issues: list[str] = []
        if report.missing_keys:
            missing = ", ".join(sorted(report.missing_keys))
            issues.append(f"unknown keys: {missing}")
        if report.conflicts:
            conflicts = ", ".join(sorted(report.conflicts))
            issues.append(f"type conflicts at: {conflicts}")
        raise ConfigValidationError(f"{label} has invalid overrides ({'; '.join(issues)})")

    return report.merged


def _merge_preset_values(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    path: tuple[str, ...],
) -> _PresetMergeReport:
    merged = dict(base)
    missing_keys: list[str] = []
    conflicts: list[str] = []

    for key, override_value in overrides.items():
        current_path = (*path, key)
        if key not in base:
            if _is_extendable_path(current_path):
                merged[key] = override_value
            else:
                missing_keys.append(".".join(current_path))
            continue

        base_value = base[key]
        if override_value is None:
            merged[key] = None
            continue

        if base_value is None:
            merged[key] = override_value
            continue

        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            report = _merge_preset_values(base_value, override_value, path=current_path)
            merged[key] = report.merged
            missing_keys.extend(report.missing_keys)
            conflicts.extend(report.conflicts)
            continue

        if isinstance(base_value, Mapping) != isinstance(override_value, Mapping):
            conflicts.append(
                f"{'.'.join(current_path)} (base {type(base_value).__name__}, "
                f"override {type(override_value).__name__})"
            )
            continue

        merged[key] = override_value

    return _PresetMergeReport(
        merged=merged,
        missing_keys=missing_keys,
        conflicts=conflicts,
    )


def _is_extendable_path(path: tuple[str, ...]) -> bool:
    for allowed_path in _EXTENDABLE_OVERRIDE_PATHS:
        if path[: len(allowed_path)] == allowed_path:
            return True
    return False


def _format_override_label(source: str, selection_name: str | None) -> str:
    if selection_name:
        return f"{source.capitalize()} preset '{selection_name}'"
    return f"{source.capitalize()} overrides"


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
    include_parameter_groups: list[str] = field(
        default_factory=lambda: list(DEFAULT_PARAMETER_GROUPS)
    )


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
        raise ConfigValidationError(f"Unknown objective preset: {name}. Available: {available}")
    return name


def _select_optimize_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    if "optimize" not in raw_config:
        return raw_config

    optimize_raw = raw_config.get("optimize")
    if optimize_raw is None:
        return {}
    if not isinstance(optimize_raw, dict):
        raise ConfigValidationError("optimize must be a mapping")

    presets_raw = optimize_raw.get("presets")
    if presets_raw is None:
        return optimize_raw
    if not isinstance(presets_raw, dict):
        raise ConfigValidationError("optimize.presets must be a mapping")
    if not presets_raw:
        raise ConfigValidationError("optimize.presets must not be empty")

    preset_name = optimize_raw.get("preset")
    if preset_name is None:
        if len(presets_raw) == 1:
            preset_name = next(iter(presets_raw))
        else:
            raise ConfigValidationError(
                "optimize.preset is required when multiple optimize presets are defined"
            )
    if not isinstance(preset_name, str):
        raise ConfigValidationError("optimize.preset must be a string")
    if preset_name not in presets_raw:
        available = ", ".join(sorted(presets_raw))
        raise ConfigValidationError(
            f"Unknown optimize preset: {preset_name}. Available: {available}"
        )

    preset_config = presets_raw[preset_name]
    if not isinstance(preset_config, dict):
        raise ConfigValidationError(f"optimize.presets.{preset_name} must be a mapping")
    return preset_config


def _parse_parameter_overrides(parameter_space_raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    overrides_raw = parameter_space_raw.get("overrides")
    if overrides_raw is None:
        return {}
    if not isinstance(overrides_raw, dict):
        raise ConfigValidationError("parameter_space.overrides must be a mapping")

    overrides: dict[str, dict[str, Any]] = {}
    for name, override_value in overrides_raw.items():
        if not isinstance(name, str):
            raise ConfigValidationError("parameter_space.overrides keys must be strings")
        if override_value is None:
            overrides[name] = {}
            continue
        if not isinstance(override_value, dict):
            raise ConfigValidationError(f"parameter_space.overrides.{name} must be a mapping")
        overrides[name] = override_value

    return overrides


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
    raw_config = resolve_optimize_preset_inheritance(raw_config)
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
    # Use a sentinel to avoid masking falsy non-mapping values.
    param_space_raw = raw_config.get("parameter_space", _PARAMETER_SPACE_MISSING)
    if param_space_raw is _PARAMETER_SPACE_MISSING:
        param_space_config: Mapping[str, Any] = {}
    elif not isinstance(param_space_raw, Mapping):
        raise ConfigValidationError("parameter_space must be a mapping")
    else:
        param_space_config = param_space_raw
    include_groups = _normalize_parameter_groups(param_space_config.get("include_groups"))
    parameter_overrides = _parse_parameter_overrides(dict(param_space_config))

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
