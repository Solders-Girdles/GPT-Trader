"""Unit tests for optimize CLI config loader parsing and overrides."""

from __future__ import annotations

import pytest

from gpt_trader.cli.commands.optimize.config_loader import (
    ConfigValidationError,
    create_default_config,
    merge_cli_overrides,
    parse_config,
)


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
