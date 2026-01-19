"""Unit tests for optimize CLI config loader file loading."""

from __future__ import annotations

import pytest

from gpt_trader.cli.commands.optimize.config_loader import (
    ConfigValidationError,
    load_config_file,
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
