"""
Consolidated configuration utilities for GPT-Trader.

This module provides centralized configuration loading, saving, and validation
functions that were previously duplicated across multiple files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigManager:
    """Centralized configuration management utilities."""

    @staticmethod
    def load_json_config(path: str | Path) -> dict[str, Any]:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file cannot be loaded or parsed
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise ValueError(f"Configuration file not found: {path}")

            with open(file_path) as f:
                config_data: dict[str, Any] = json.load(f)
                return config_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading configuration from {path}: {e}") from e

    @staticmethod
    def save_json_config(config: dict[str, Any], path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            config: Configuration dictionary to save
            path: Path where to save the configuration

        Raises:
            ValueError: If file cannot be saved
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        except Exception as e:
            raise ValueError(f"Error saving configuration to {path}: {e}") from e

    @staticmethod
    def load_yaml_config(path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file cannot be loaded or parsed
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise ValueError(f"Configuration file not found: {path}")

            with open(file_path) as f:
                return yaml.safe_load(f) or {}

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading configuration from {path}: {e}") from e

    @staticmethod
    def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            path: Path where to save the configuration

        Raises:
            ValueError: If file cannot be saved
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                yaml.safe_dump(config, f, indent=2, default_flow_style=False)

        except Exception as e:
            raise ValueError(f"Error saving configuration to {path}: {e}") from e

    @staticmethod
    def merge_configs(
        base_config: dict[str, Any], override_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration values to override base

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def validate_required_keys(config: dict[str, Any], required_keys: list[str]) -> None:
        """Validate configuration has required keys.

        Args:
            config: Configuration dictionary
            required_keys: List of required key names

        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"Configuration missing required keys: {missing_keys}")

    @staticmethod
    def get_nested_value(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'database.host')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @staticmethod
    def set_nested_value(config: dict[str, Any], key_path: str, value: Any) -> None:
        """Set nested configuration value using dot notation.

        Args:
            config: Configuration dictionary to modify
            key_path: Dot-separated key path (e.g., 'database.host')
            value: Value to set
        """
        keys = key_path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value


class DefaultConfigs:
    """Default configuration templates."""

    @staticmethod
    def get_trading_defaults() -> dict[str, Any]:
        """Get default trading configuration."""
        return {
            "risk": {"max_portfolio_risk": 0.02, "max_position_risk": 0.005, "stop_loss_pct": 0.05},
            "execution": {
                "slippage_bps": 5.0,
                "commission_per_share": 0.01,
                "min_position_size": 100,
            },
            "strategy": {
                "rebalance_frequency": "daily",
                "lookback_days": 252,
                "min_trade_interval": 1,
            },
        }

    @staticmethod
    def get_backtest_defaults() -> dict[str, Any]:
        """Get default backtesting configuration."""
        return {
            "data": {"start_date": "2020-01-01", "end_date": "2023-12-31", "frequency": "daily"},
            "execution": {
                "initial_capital": 100000,
                "slippage_model": "linear",
                "commission_model": "fixed",
            },
            "reporting": {"create_plots": True, "save_trades": True, "benchmark": "SPY"},
        }

    @staticmethod
    def get_optimization_defaults() -> dict[str, Any]:
        """Get default optimization configuration."""
        return {
            "method": "grid",
            "max_workers": 4,
            "early_stopping": True,
            "patience": 20,
            "min_improvement": 0.001,
            "cross_validation": {"enabled": True, "folds": 5, "purge_days": 7},
        }
