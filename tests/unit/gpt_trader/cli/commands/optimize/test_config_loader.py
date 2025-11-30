"""Unit tests for optimize CLI config loader."""

from __future__ import annotations

from datetime import datetime

import pytest

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


class TestLoadConfigFile:
    def test_loads_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("study:\n  name: test_study\n")

        result = load_config_file(config_file)
        assert result["study"]["name"] == "test_study"

    def test_raises_for_missing_file(self, tmp_path):
        """Test raises error for missing file."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(ConfigValidationError, match="not found"):
            load_config_file(config_file)

    def test_raises_for_invalid_yaml(self, tmp_path):
        """Test raises error for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("key: value: invalid")

        with pytest.raises(ConfigValidationError, match="Invalid YAML"):
            load_config_file(config_file)


class TestParseConfig:
    def test_parses_minimal_config(self):
        """Test parsing minimal configuration."""
        raw = {"study": {"name": "test"}}
        config = parse_config(raw)

        assert config.study.name == "test"
        assert config.study.trials == 100  # default
        assert config.objective_name == "sharpe"  # default

    def test_parses_full_config(self):
        """Test parsing full configuration."""
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

    def test_raises_for_missing_study_name(self):
        """Test raises error for missing study name."""
        raw = {"study": {}}

        with pytest.raises(ConfigValidationError, match="study.name is required"):
            parse_config(raw)


class TestMergeCliOverrides:
    def test_overrides_trials(self):
        """Test CLI overrides trials."""
        config = create_default_config("test")
        original_trials = config.study.trials

        result = merge_cli_overrides(config, {"trials": 50})

        assert result.study.trials == 50
        assert result.study.trials != original_trials

    def test_overrides_objective(self):
        """Test CLI overrides objective."""
        config = create_default_config("test")

        result = merge_cli_overrides(config, {"objective": "sortino"})

        assert result.objective_name == "sortino"

    def test_overrides_symbols(self):
        """Test CLI overrides symbols."""
        config = create_default_config("test")

        result = merge_cli_overrides(config, {"symbols": ["ETH-USD"]})

        assert result.symbols == ["ETH-USD"]


class TestCreateObjectiveFromPreset:
    def test_creates_simple_objective(self):
        """Test creating simple objective."""
        objective = create_objective_from_preset("sharpe")
        assert objective.name == "sharpe_ratio"

    def test_creates_composite_objective(self):
        """Test creating composite objective with factory."""
        objective = create_objective_from_preset("risk_averse")
        assert objective.name == "risk_averse"

    def test_passes_kwargs_to_factory(self):
        """Test kwargs are passed to factory."""
        objective = create_objective_from_preset("risk_averse", max_drawdown_pct=10.0)
        # Find drawdown constraint
        constraint_names = [c.name for c in objective.constraints]
        assert "max_drawdown" in constraint_names

    def test_raises_for_unknown_preset(self):
        """Test raises error for unknown preset."""
        with pytest.raises(ConfigValidationError, match="Unknown objective"):
            create_objective_from_preset("nonexistent")


class TestGetObjectiveDirection:
    def test_maximize_presets(self):
        """Test maximize direction presets."""
        assert get_objective_direction("sharpe") == "maximize"
        assert get_objective_direction("sortino") == "maximize"
        assert get_objective_direction("total_return") == "maximize"

    def test_minimize_presets(self):
        """Test minimize direction presets."""
        assert get_objective_direction("max_drawdown") == "minimize"

    def test_raises_for_unknown_preset(self):
        """Test raises error for unknown preset."""
        with pytest.raises(ConfigValidationError):
            get_objective_direction("nonexistent")


class TestBuildParameterSpaceFromConfig:
    def test_includes_strategy_parameters(self):
        """Test includes strategy parameters."""
        config = create_default_config("test")
        config.include_parameter_groups = ["strategy"]

        space = build_parameter_space_from_config(config)

        assert len(space.strategy_parameters) > 0
        assert len(space.risk_parameters) == 0

    def test_includes_risk_parameters(self):
        """Test includes risk parameters."""
        config = create_default_config("test")
        config.include_parameter_groups = ["risk"]

        space = build_parameter_space_from_config(config)

        assert len(space.risk_parameters) > 0

    def test_includes_all_groups(self):
        """Test includes all parameter groups."""
        config = create_default_config("test")
        config.include_parameter_groups = ["strategy", "risk", "simulation"]

        space = build_parameter_space_from_config(config)

        assert len(space.strategy_parameters) > 0
        assert len(space.risk_parameters) > 0
        assert len(space.simulation_parameters) > 0


class TestBuildOptimizationConfig:
    def test_builds_valid_config(self):
        """Test building valid optimization config."""
        cli_config = create_default_config("test_study")
        cli_config.include_parameter_groups = ["strategy"]

        opt_config = build_optimization_config(cli_config)

        assert opt_config.study_name == "test_study"
        assert opt_config.number_of_trials == 100
        assert opt_config.direction == "maximize"


class TestCreateDefaultConfig:
    def test_creates_with_minimal_args(self):
        """Test creating default config with minimal arguments."""
        config = create_default_config("test")

        assert config.study.name == "test"
        assert config.objective_name == "sharpe"
        assert config.symbols == ["BTC-USD"]

    def test_creates_with_all_args(self):
        """Test creating default config with all arguments."""
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


class TestObjectivePresets:
    def test_all_presets_have_direction(self):
        """Test all presets have a direction defined."""
        for preset_name in OBJECTIVE_PRESETS:
            factory, direction = OBJECTIVE_PRESETS[preset_name]
            assert direction in ("maximize", "minimize")

    def test_all_presets_can_be_created(self):
        """Test all presets can be instantiated."""
        for preset_name in OBJECTIVE_PRESETS:
            objective = create_objective_from_preset(preset_name)
            assert objective is not None
