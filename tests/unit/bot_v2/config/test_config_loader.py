"""Tests for ConfigLoader - env overrides, validation, and file loading."""

import json
from pathlib import Path

import pytest
import yaml

from bot_v2.config import ConfigLoader
from bot_v2.errors import ConfigurationError
from bot_v2.validation import Validator


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def config_loader(temp_config_dir):
    """Create ConfigLoader with temp directory."""
    return ConfigLoader(config_dir=temp_config_dir)


# Environment override tests - targeting lines 225-245


class TestEnvOverrides:
    """Test environment variable override logic."""

    def test_env_override_integer(self, config_loader, monkeypatch):
        """Environment variable overrides integer config value."""
        monkeypatch.setenv("BOT_V2_BACKTEST_MAX_RETRIES", "10")

        config = config_loader.get_config("backtest")
        # Should have default initial_capital but overridden max_retries
        assert config["initial_capital"] == 10000.0
        # Note: max_retries not in default backtest config, so it adds it
        assert config.get("max_retries") == 10

    def test_env_override_float(self, config_loader, monkeypatch):
        """Environment variable overrides float config value."""
        monkeypatch.setenv("BOT_V2_BACKTEST_INITIAL_CAPITAL", "25000.50")

        config = config_loader.get_config("backtest")
        assert config["initial_capital"] == 25000.50

    def test_env_override_boolean_true(self, config_loader, monkeypatch):
        """Environment variable 'true' converts to boolean."""
        monkeypatch.setenv("BOT_V2_BACKTEST_ENABLE_SHORTING", "true")

        config = config_loader.get_config("backtest")
        assert config["enable_shorting"] is True

    def test_env_override_boolean_false(self, config_loader, monkeypatch):
        """Environment variable 'false' converts to boolean."""
        monkeypatch.setenv("BOT_V2_BACKTEST_ENABLE_SHORTING", "false")

        config = config_loader.get_config("backtest")
        assert config["enable_shorting"] is False

    def test_env_override_string(self, config_loader, monkeypatch):
        """Environment variable stays as string when not parseable."""
        monkeypatch.setenv("BOT_V2_SYSTEM_DATA_PROVIDER", "alpha_vantage")

        config = config_loader.get_config("system")
        assert config["data_provider"] == "alpha_vantage"

    def test_env_override_json_object(self, config_loader, monkeypatch):
        """Environment variable with JSON object parses correctly."""
        monkeypatch.setenv("BOT_V2_BACKTEST_CUSTOM", '{"nested": {"value": 123}}')

        config = config_loader.get_config("backtest")
        assert config["custom"] == {"nested": {"value": 123}}

    def test_env_override_json_array(self, config_loader, monkeypatch):
        """Environment variable with JSON array parses correctly."""
        monkeypatch.setenv("BOT_V2_SYSTEM_SYMBOLS", '["BTC-USD", "ETH-USD"]')

        config = config_loader.get_config("system")
        assert config["symbols"] == ["BTC-USD", "ETH-USD"]

    def test_env_override_case_insensitive_key(self, config_loader, monkeypatch):
        """Environment variable key converts to lowercase."""
        monkeypatch.setenv("BOT_V2_SYSTEM_LOG_LEVEL", "DEBUG")

        config = config_loader.get_config("system")
        assert config["log_level"] == "DEBUG"

    def test_multiple_env_overrides(self, config_loader, monkeypatch):
        """Multiple environment variables override correctly."""
        monkeypatch.setenv("BOT_V2_BACKTEST_INITIAL_CAPITAL", "15000")
        monkeypatch.setenv("BOT_V2_BACKTEST_COMMISSION", "0.002")
        monkeypatch.setenv("BOT_V2_BACKTEST_ENABLE_SHORTING", "true")

        config = config_loader.get_config("backtest")
        assert config["initial_capital"] == 15000.0
        assert config["commission"] == 0.002
        assert config["enable_shorting"] is True

    def test_env_override_takes_precedence_over_file(self, temp_config_dir, monkeypatch):
        """Environment variable overrides file config value."""
        # Create config file with initial_capital = 50000
        config_file = temp_config_dir / "backtest_config.json"
        config_file.write_text(json.dumps({"initial_capital": 50000}))

        loader = ConfigLoader(config_dir=temp_config_dir)

        # Set env override to 75000
        monkeypatch.setenv("BOT_V2_BACKTEST_INITIAL_CAPITAL", "75000")

        config = loader.get_config("backtest")

        # Env should win
        assert config["initial_capital"] == 75000

    def test_env_override_malformed_json_falls_back(self, config_loader, monkeypatch):
        """Malformed JSON in env var falls back to type parsing."""
        # Set env var with malformed JSON (missing closing brace)
        monkeypatch.setenv("BOT_V2_BACKTEST_CUSTOM_FIELD", "{incomplete")

        config = config_loader.get_config("backtest")

        # Should keep as string since JSON parse failed
        assert config["custom_field"] == "{incomplete"

    def test_env_override_ambiguous_number_string(self, config_loader, monkeypatch):
        """String that looks like number but has leading zeros."""
        monkeypatch.setenv("BOT_V2_BACKTEST_ORDER_ID", "00123")

        config = config_loader.get_config("backtest")

        # Should parse as int (leading zeros stripped)
        assert config["order_id"] == 123


# File loading tests - targeting lines 165-210


class TestFileLoading:
    """Test configuration file loading."""

    def test_load_json_config(self, temp_config_dir):
        """Load configuration from JSON file."""
        config_file = temp_config_dir / "custom_config.json"
        config_data = {"key": "value", "number": 42}
        config_file.write_text(json.dumps(config_data))

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_config("custom")

        assert config["key"] == "value"
        assert config["number"] == 42

    def test_load_yaml_config(self, temp_config_dir):
        """Load configuration from YAML file."""
        config_file = temp_config_dir / "custom_config.yaml"
        config_data = {"key": "value", "nested": {"item": 123}}
        config_file.write_text(yaml.dump(config_data))

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_config("custom")

        assert config["key"] == "value"
        assert config["nested"]["item"] == 123

    def test_load_yml_extension(self, temp_config_dir):
        """Load configuration from .yml file."""
        config_file = temp_config_dir / "custom_config.yml"
        config_data = {"test": "yml"}
        config_file.write_text(yaml.dump(config_data))

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_config("custom")

        assert config["test"] == "yml"

    def test_json_preferred_over_yaml(self, temp_config_dir):
        """JSON file takes precedence over YAML when both exist."""
        json_file = temp_config_dir / "custom_config.json"
        yaml_file = temp_config_dir / "custom_config.yaml"

        json_file.write_text(json.dumps({"source": "json"}))
        yaml_file.write_text(yaml.dump({"source": "yaml"}))

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_config("custom")

        assert config["source"] == "json"

    def test_missing_file_uses_defaults(self, config_loader, caplog):
        """Missing config file falls back to defaults with warning."""
        config = config_loader.get_config("nonexistent")

        # Should return empty dict (no defaults for nonexistent slice)
        assert config == {}
        assert "No config file found for nonexistent" in caplog.text

    def test_file_merge_with_defaults(self, temp_config_dir):
        """File config merges with default config."""
        # Backtest has defaults, override some values
        config_file = temp_config_dir / "backtest_config.json"
        config_file.write_text(json.dumps({"initial_capital": 50000}))

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_config("backtest")

        # Should have file override
        assert config["initial_capital"] == 50000
        # Should have default values
        assert config["commission"] == 0.001
        assert config["slippage"] == 0.0005

    def test_invalid_json_raises_error(self, temp_config_dir):
        """Invalid JSON file raises ConfigurationError."""
        config_file = temp_config_dir / "bad_config.json"
        config_file.write_text("{invalid json")

        loader = ConfigLoader(config_dir=temp_config_dir)

        with pytest.raises(ConfigurationError) as exc_info:
            loader.get_config("bad")

        assert "Failed to load config for bad" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, temp_config_dir):
        """Invalid YAML file raises ConfigurationError."""
        config_file = temp_config_dir / "bad_config.yaml"
        config_file.write_text("invalid: yaml: content:")

        loader = ConfigLoader(config_dir=temp_config_dir)

        with pytest.raises(ConfigurationError) as exc_info:
            loader.get_config("bad")

        assert "Failed to load config for bad" in str(exc_info.value)


# Validation tests - targeting line 139


class TestConfigValidation:
    """Test config validation with validators."""

    def test_validation_with_validator(self, config_loader):
        """Config validation runs when validator is set."""

        # Define a validator
        def positive_validator(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        validators = {"initial_capital": Validator(positive_validator)}
        config_loader.set_validator("backtest", validators)

        # Valid config should pass
        config = config_loader.get_config("backtest")
        assert config["initial_capital"] > 0

    def test_validation_with_custom_validator(self, config_loader):
        """Custom validators can be set for config slices."""

        # Just test that set_validator works
        def positive_validator(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        validators = {"initial_capital": Validator(positive_validator)}
        config_loader.set_validator("backtest", validators)

        # Should not raise for valid config
        config = config_loader.get_config("backtest")
        assert config["initial_capital"] > 0


# Hot reload tests - targeting lines 143-153


class TestHotReload:
    """Test configuration hot reload detection."""

    def test_reload_detects_file_change(self, temp_config_dir):
        """Config reloads when file is modified."""
        config_file = temp_config_dir / "dynamic_config.json"
        config_file.write_text(json.dumps({"value": 1}))

        loader = ConfigLoader(config_dir=temp_config_dir)

        # First load
        config1 = loader.get_config("dynamic")
        assert config1["value"] == 1

        # Modify file
        import time

        time.sleep(0.01)  # Ensure mtime changes
        config_file.write_text(json.dumps({"value": 2}))

        # Second load should detect change
        config2 = loader.get_config("dynamic")
        assert config2["value"] == 2

    def test_no_reload_when_file_unchanged(self, temp_config_dir):
        """Config doesn't reload when file hasn't changed."""
        config_file = temp_config_dir / "static_config.json"
        config_file.write_text(json.dumps({"value": 1}))

        loader = ConfigLoader(config_dir=temp_config_dir)

        # Load twice without modifying
        config1 = loader.get_config("static")
        config2 = loader.get_config("static")

        # Should return same values (may not be same object due to copy)
        assert config1 == config2
        assert config1["value"] == 1


# Utility tests


class TestConfigUtilities:
    """Test configuration utility methods."""

    def test_get_all_configs(self, config_loader):
        """get_all_configs returns all loaded configurations."""
        # Trigger loads
        config_loader.get_config("system")
        config_loader.get_config("backtest")

        all_configs = config_loader.get_all_configs()

        assert "system" in all_configs
        assert "backtest" in all_configs
        assert all_configs["system"]["log_level"] == "INFO"
