"""
Enhanced settings management for GPT-Trader.

This module provides a robust configuration management system with
validation, environment variable handling, and multiple configuration sources.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from bot.utils.base import BaseConfig
from bot.utils.config import ConfigManager
from pydantic import BaseModel

T = TypeVar("T", bound="EnhancedSettings")


class EnhancedSettings(BaseConfig):
    """Enhanced settings base class with multiple configuration sources."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def save_config(self, path: str | Path, format: str = "json") -> None:
        """Save current configuration to file.

        Args:
            path: Path where to save configuration
            format: Format to save in ('json', 'yaml')
        """
        config_dict = self.model_dump(exclude_unset=False)

        if format.lower() == "json":
            ConfigManager.save_json_config(config_dict, path)
        elif format.lower() == "yaml":
            ConfigManager.save_yaml_config(config_dict, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load_config(cls: type[T], path: str | Path, format: str | None = None) -> T:
        """Load configuration from file.

        Args:
            path: Path to configuration file
            format: Format to load ('json', 'yaml', auto-detect if None)

        Returns:
            Settings instance
        """
        file_path = Path(path)

        if format is None:
            format = file_path.suffix.lower().lstrip(".")

        if format == "json":
            config_dict = ConfigManager.load_json_config(file_path)
        elif format in ("yaml", "yml"):
            config_dict = ConfigManager.load_yaml_config(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return cls(**config_dict)

    def merge_config(self: T, other: dict[str, Any] | T) -> T:
        """Merge with another configuration.

        Args:
            other: Dictionary or settings instance to merge

        Returns:
            New settings instance with merged values
        """
        base_dict = self.model_dump()
        other_dict = other.model_dump() if isinstance(other, BaseModel) else other

        merged_dict = ConfigManager.merge_configs(base_dict, other_dict)
        return self.__class__.model_validate(merged_dict)

    def validate_config(self) -> list[str]:
        """Validate configuration and return any issues.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Validate required environment variables
        required_env_vars = self._get_required_env_vars()
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                issues.append(f"Required environment variable not set: {env_var}")

        # Validate file paths exist
        file_paths = self._get_file_path_fields()
        for field_name, file_path in file_paths.items():
            if file_path and not Path(file_path).exists():
                issues.append(f"File path does not exist: {field_name} = {file_path}")

        return issues

    def _get_required_env_vars(self) -> list[str]:
        """Get list of required environment variables.

        Override in subclasses to specify required env vars.
        """
        return []

    def _get_file_path_fields(self) -> dict[str, Path | str | None]:
        """Get file path fields for validation.

        Override in subclasses to specify file path fields.
        """
        return {}

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary (sensitive values masked).

        Returns:
            Configuration summary with sensitive data masked
        """
        config = self.model_dump()
        return self._mask_sensitive_values(config)

    def _mask_sensitive_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive configuration values."""
        sensitive_keys = {
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "secret_key",
            "private_key",
            "auth_token",
        }

        masked_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                masked_config[key] = self._mask_sensitive_values(value)
            elif any(sensitive_word in key.lower() for sensitive_word in sensitive_keys):
                masked_config[key] = "***MASKED***" if value else None
            else:
                masked_config[key] = value

        return masked_config


class ConfigurationValidator:
    """Validates configuration across multiple sources."""

    def __init__(self, settings: EnhancedSettings) -> None:
        self.settings = settings

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all configuration aspects.

        Returns:
            Dictionary of validation results by category
        """
        results = {
            "environment": self._validate_environment_variables(),
            "files": self._validate_file_paths(),
            "connectivity": self._validate_connectivity(),
            "permissions": self._validate_permissions(),
            "resources": self._validate_resource_limits(),
        }

        return results

    def _validate_environment_variables(self) -> list[str]:
        """Validate environment variables."""
        issues = []

        # Check for common environment variable issues
        env_vars = os.environ

        # Check for conflicting environment settings
        if env_vars.get("DEBUG") == "true" and env_vars.get("ENVIRONMENT") == "production":
            issues.append("DEBUG mode enabled in production environment")

        return issues

    def _validate_file_paths(self) -> list[str]:
        """Validate file paths in configuration."""
        issues = []

        # Get file path fields from settings
        file_paths = self.settings._get_file_path_fields()

        for field_name, file_path in file_paths.items():
            if file_path:
                path_obj = Path(file_path)

                if not path_obj.exists():
                    issues.append(f"Path does not exist: {field_name} = {file_path}")
                elif path_obj.is_file():
                    # Check file permissions
                    if not os.access(path_obj, os.R_OK):
                        issues.append(f"File not readable: {file_path}")
                elif path_obj.is_dir():
                    # Check directory permissions
                    if not os.access(path_obj, os.W_OK):
                        issues.append(f"Directory not writable: {file_path}")

        return issues

    def _validate_connectivity(self) -> list[str]:
        """Validate network connectivity requirements."""
        issues: list[str] = []

        # This would test connectivity to external services
        # For now, just check if required URLs are properly formatted

        return issues

    def _validate_permissions(self) -> list[str]:
        """Validate file and directory permissions."""
        issues: list[str] = []

        # Check common directories that need to be writable
        writable_dirs = ["data", "logs", "cache"]

        for dir_name in writable_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and not os.access(dir_path, os.W_OK):
                issues.append(f"Directory not writable: {dir_name}")

        return issues

    def _validate_resource_limits(self) -> list[str]:
        """Validate resource limits and constraints."""
        issues = []

        # Check system resources
        try:
            import psutil

            # Check available memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # psutil not available, skip resource validation
            return issues
        if hasattr(self.settings, "optimization") and hasattr(
            self.settings.optimization, "memory_limit_gb"
        ):
            required_memory = self.settings.optimization.memory_limit_gb
            if available_memory_gb < required_memory:
                issues.append(
                    f"Insufficient memory: {available_memory_gb:.1f}GB available, {required_memory}GB required"
                )

        # Check CPU count
        cpu_count = os.cpu_count() or 1
        if hasattr(self.settings, "optimization") and hasattr(
            self.settings.optimization, "max_workers"
        ):
            max_workers = self.settings.optimization.max_workers
            if max_workers > cpu_count * 2:  # Allow 2x CPU count as reasonable upper limit
                issues.append(
                    f"Too many workers configured: {max_workers} workers on {cpu_count} CPUs"
                )

        return issues


class ConfigurationManager:
    """Centralized configuration management."""

    def __init__(self, settings: EnhancedSettings) -> None:
        self.settings = settings
        self.validator = ConfigurationValidator(settings)

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive configuration health check.

        Returns:
            Health check results
        """
        validation_results = self.validator.validate_all()

        total_issues = sum(len(issues) for issues in validation_results.values())

        return {
            "status": "healthy" if total_issues == 0 else "unhealthy",
            "total_issues": total_issues,
            "validation_results": validation_results,
            "config_summary": self.settings.get_config_summary(),
            "timestamp": str(pd.Timestamp.now()),
        }

    def export_config(self, path: str | Path, include_env_vars: bool = False) -> None:
        """Export current configuration to file.

        Args:
            path: Export path
            include_env_vars: Whether to include environment variables
        """
        config_data = {
            "settings": self.settings.model_dump(),
            "validation_results": self.validator.validate_all(),
        }

        if include_env_vars:
            # Include non-sensitive environment variables
            config_data["environment"] = {
                key: value
                for key, value in os.environ.items()
                if not any(
                    sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]
                )
            }

        ConfigManager.save_json_config(config_data, path)
