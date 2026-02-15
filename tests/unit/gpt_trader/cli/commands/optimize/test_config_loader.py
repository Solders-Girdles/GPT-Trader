"""Unit tests for optimize CLI config loader."""

from __future__ import annotations

from datetime import datetime

import pytest

from gpt_trader.cli.commands.optimize.config_loader import (
    ConfigValidationError,
    build_optimization_config,
    build_parameter_space_from_config,
    create_default_config,
    create_objective_from_preset,
    get_objective_direction,
    load_config_file,
    merge_cli_overrides,
    parse_config,
    resolve_optimize_preset_inheritance,
)
from gpt_trader.cli.commands.optimize.registry import list_objective_names


def test_load_config_file_loads_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("study:\n  name: test_study\n")

    result = load_config_file(config_file)

    assert result["study"]["name"] == "test_study"


def test_load_config_file_missing_file(tmp_path):
    config_file = tmp_path / "missing.yaml"

    with pytest.raises(ConfigValidationError, match="not found"):
        load_config_file(config_file)


def test_load_config_file_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: invalid")

    with pytest.raises(ConfigValidationError, match="Invalid YAML"):
        load_config_file(config_file)


def test_parse_config_minimal():
    raw = {"study": {"name": "test"}}

    config = parse_config(raw)

    assert config.study.name == "test"
    assert config.study.trials == 100
    assert config.objective_name == "sharpe"


def test_parse_config_full():
    raw = {
        "study": {
            "name": "full_study",
            "trials": 200,
            "sampler": "cmaes",
        },
        "objective": {"preset": "risk_averse"},
        "strategy": {"type": "perps_baseline", "symbols": ["BTC-USD", "ETH-USD"]},
        "backtest": {
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "granularity": "ONE_HOUR",
        },
    }

    config = parse_config(raw)

    assert config.study.name == "full_study"
    assert config.study.trials == 200
    assert config.study.sampler == "cmaes"
    assert config.objective_name == "risk_averse"
    assert config.symbols == ["BTC-USD", "ETH-USD"]
    assert config.backtest.granularity == "ONE_HOUR"


def test_resolve_optimize_preset_inheritance_merges_precedence():
    raw = {
        "presets": {
            "base": {
                "study": {"name": "base", "trials": 100},
                "objective": {"preset": "sharpe"},
            },
            "profiles": {
                "dev": {
                    "study": {"trials": 200},
                    "objective": {"preset": "sortino"},
                }
            },
            "scenarios": {"fast": {"study": {"trials": 25}}},
        },
        "profile": "dev",
        "scenario": "fast",
    }

    resolved = resolve_optimize_preset_inheritance(raw)

    assert resolved["study"]["name"] == "base"
    assert resolved["study"]["trials"] == 25
    assert resolved["objective"]["preset"] == "sortino"


def test_resolve_optimize_preset_inheritance_ignores_root_profile_for_optimize_container():
    raw = {
        "profile": "root_profile",
        "optimize": {
            "presets": {
                "base": {"study": {"name": "base", "trials": 100}},
                "profiles": {
                    "root_profile": {"study": {"trials": 1}},
                    "optimize_profile": {"study": {"trials": 25}},
                },
            },
            "profile": "optimize_profile",
        },
    }

    resolved = resolve_optimize_preset_inheritance(raw)

    assert resolved["study"]["trials"] == 25


def test_resolve_optimize_preset_inheritance_rejects_falsy_non_mapping_override():
    raw = {
        "presets": {
            "base": {"study": {"name": "base", "trials": 100}},
            "profiles": {"broken": []},
        },
        "profile": "broken",
    }

    with pytest.raises(
        ConfigValidationError,
        match="profile preset 'broken' overrides must be a mapping",
    ):
        resolve_optimize_preset_inheritance(raw)


def test_resolve_optimize_preset_inheritance_keeps_empty_overrides():
    raw = {
        "presets": {
            "base": {"study": {"name": "base", "trials": 100}},
            "scenarios": {"empty": {"study": {}}},
        },
        "scenario": "empty",
    }

    resolved = resolve_optimize_preset_inheritance(raw)

    assert resolved["study"]["name"] == "base"
    assert resolved["study"]["trials"] == 100


def test_resolve_optimize_preset_inheritance_preserves_explicit_nulls():
    raw = {
        "presets": {
            "base": {"objective": {"preset": "sharpe"}},
            "scenarios": {"nulls": {"objective": {"preset": None}}},
        },
        "scenario": "nulls",
    }

    resolved = resolve_optimize_preset_inheritance(raw)

    assert "objective" in resolved
    assert resolved["objective"]["preset"] is None


def test_resolve_optimize_preset_inheritance_unknown_override_key():
    raw = {
        "presets": {
            "base": {"study": {"name": "base", "trials": 100}},
            "scenarios": {"broken": {"study": {"trialz": 10}}},
        },
        "scenario": "broken",
    }

    with pytest.raises(ConfigValidationError, match="unknown keys: study\\.trialz"):
        resolve_optimize_preset_inheritance(raw)


def test_resolve_optimize_preset_inheritance_type_conflict():
    raw = {
        "presets": {
            "base": {"parameter_space": {"include_groups": ["strategy"]}},
            "scenarios": {"broken": {"parameter_space": ["strategy"]}},
        },
        "scenario": "broken",
    }

    with pytest.raises(ConfigValidationError, match="type conflicts at: parameter_space"):
        resolve_optimize_preset_inheritance(raw)


def test_parse_config_invalid_parameter_override_type():
    raw = {
        "study": {"name": "override_test"},
        "parameter_space": {"overrides": {"short_ma_period": 0}},
    }

    with pytest.raises(ConfigValidationError, match="parameter_space.overrides.short_ma_period"):
        parse_config(raw)


def test_parse_config_missing_study_name():
    with pytest.raises(ConfigValidationError, match="study.name is required"):
        parse_config({"study": {}})


def test_parse_config_invalid_parameter_group():
    raw = {
        "study": {"name": "group_test"},
        "parameter_space": {"include_groups": ["strategy", "unknown_group"]},
    }

    with pytest.raises(ConfigValidationError, match="Unknown parameter group"):
        parse_config(raw)


@pytest.mark.parametrize("parameter_space", ["", [], 0, None])
def test_parse_config_invalid_parameter_space_type(parameter_space):
    raw = {
        "study": {"name": "param_space_type"},
        "parameter_space": parameter_space,
    }

    with pytest.raises(ConfigValidationError, match="parameter_space must be a mapping"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("overrides", "expected_trials", "expected_objective", "expected_symbols"),
    [
        ({"trials": 50}, 50, "sharpe", ["BTC-USD"]),
        ({"objective": "sortino"}, 100, "sortino", ["BTC-USD"]),
        ({"symbols": ["ETH-USD"]}, 100, "sharpe", ["ETH-USD"]),
    ],
)
def test_merge_cli_overrides(overrides, expected_trials, expected_objective, expected_symbols):
    config = create_default_config("test")

    result = merge_cli_overrides(config, overrides)

    assert result.study.trials == expected_trials
    assert result.objective_name == expected_objective
    assert result.symbols == expected_symbols


def test_create_default_config_minimal():
    config = create_default_config("test")

    assert config.study.name == "test"
    assert config.objective_name == "sharpe"
    assert config.symbols == ["BTC-USD"]


def test_create_default_config_full():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)

    config = create_default_config(
        study_name="full_test",
        objective="sortino",
        trials=50,
        symbols=["ETH-USD"],
        start_date=start,
        end_date=end,
    )

    assert config.study.name == "full_test"
    assert config.study.trials == 50
    assert config.objective_name == "sortino"
    assert config.symbols == ["ETH-USD"]
    assert config.backtest.start_date == start
    assert config.backtest.end_date == end


@pytest.mark.parametrize(
    ("groups", "expect_strategy", "expect_risk", "expect_simulation"),
    [
        (["strategy"], True, False, False),
        (["risk"], False, True, False),
        (["strategy", "risk", "simulation"], True, True, True),
    ],
)
def test_build_parameter_space_from_config(
    groups,
    expect_strategy,
    expect_risk,
    expect_simulation,
):
    config = create_default_config("test")
    config.include_parameter_groups = groups

    space = build_parameter_space_from_config(config)

    assert (len(space.strategy_parameters) > 0) is expect_strategy
    assert (len(space.risk_parameters) > 0) is expect_risk
    assert (len(space.simulation_parameters) > 0) is expect_simulation


def test_build_optimization_config():
    cli_config = create_default_config("test_study")
    cli_config.include_parameter_groups = ["strategy"]

    opt_config = build_optimization_config(cli_config)

    assert opt_config.study_name == "test_study"
    assert opt_config.number_of_trials == 100
    assert opt_config.direction == "maximize"


def test_create_objective_from_preset_simple():
    objective = create_objective_from_preset("sharpe")

    assert objective.name == "sharpe_ratio"


def test_create_objective_from_preset_composite():
    objective = create_objective_from_preset("risk_averse")

    assert objective.name == "risk_averse"


def test_create_objective_from_preset_kwargs():
    objective = create_objective_from_preset("risk_averse", max_drawdown_pct=10.0)
    constraint_names = [constraint.name for constraint in objective.constraints]

    assert "max_drawdown" in constraint_names


def test_create_objective_from_preset_unknown():
    with pytest.raises(ConfigValidationError, match="Unknown objective"):
        create_objective_from_preset("nonexistent")


@pytest.mark.parametrize("preset", ["sharpe", "sortino", "total_return"])
def test_get_objective_direction_maximize(preset):
    assert get_objective_direction(preset) == "maximize"


def test_get_objective_direction_minimize():
    assert get_objective_direction("max_drawdown") == "minimize"


def test_get_objective_direction_unknown():
    with pytest.raises(ConfigValidationError):
        get_objective_direction("nonexistent")


def test_objective_registry_has_valid_direction():
    for preset_name in list_objective_names():
        assert get_objective_direction(preset_name) in ("maximize", "minimize")


def test_objective_registry_creates_instances():
    for preset_name in list_objective_names():
        objective = create_objective_from_preset(preset_name)
        assert objective is not None
