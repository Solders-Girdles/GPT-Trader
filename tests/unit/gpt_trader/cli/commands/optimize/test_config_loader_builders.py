"""Unit tests for optimize CLI config loader builders."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.cli.commands.optimize.config_loader import (
    build_optimization_config,
    build_parameter_space_from_config,
    create_default_config,
)


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
