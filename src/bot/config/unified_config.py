"""Unified configuration system for GPT-Trader.

This module provides a single, comprehensive configuration system that:
- Consolidates all configuration sources
- Eliminates duplicate configuration classes
- Provides validation and type safety
- Supports environment variable overrides
- Enables configuration profiles (dev, test, prod)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from bot.config.financial_config import FinancialConfig
from bot.security.secrets_manager import get_secret_manager
from pydantic import BaseModel, Field, field_validator, model_validator


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DataConfig(BaseModel):
    """Data source and validation configuration."""

    cache_dir: Path = Field(
        default=Path("data/cache"), description="Directory for caching market data"
    )

    universe_file: Path = Field(
        default=Path("data/universe/universe.csv"), description="Path to universe file"
    )

    strict_validation: bool = Field(
        default=True, description="Whether to strictly validate OHLC data"
    )

    repair_data: bool = Field(default=True, description="Whether to attempt repairing invalid data")

    default_source: str = Field(
        default="yfinance", description="Default data source (yfinance, alpaca, etc.)"
    )

    max_cache_age_days: int = Field(
        default=7, description="Maximum age of cached data in days", gt=0
    )

    batch_size: int = Field(default=100, description="Batch size for data operations", gt=0)

    @field_validator("default_source")
    @classmethod
    def validate_data_source(cls, v: str) -> str:
        """Validate data source is supported."""
        valid_sources = {"yfinance", "alpaca", "polygon", "local"}
        if v not in valid_sources:
            raise ValueError(f"Invalid data source: {v}. Must be one of {valid_sources}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    file_path: Path | None = Field(
        default=None, description="Log file path (if None, logs to console only)"
    )

    max_size_mb: int = Field(default=10, description="Maximum log file size in MB", gt=0)

    backup_count: int = Field(default=5, description="Number of backup log files to keep", ge=0)

    structured_logging: bool = Field(default=False, description="Enable structured JSON logging")

    log_trades: bool = Field(default=True, description="Log all trade events")

    log_performance: bool = Field(default=True, description="Log performance metrics")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @property
    def max_size_bytes(self) -> int:
        """Get max size in bytes."""
        return self.max_size_mb * 1024 * 1024


class AlpacaConfig(BaseModel):
    """Alpaca API configuration."""

    api_key_id: str | None = Field(default=None, description="Alpaca API key ID")

    api_secret_key: str | None = Field(default=None, description="Alpaca API secret key")

    paper_base_url: str = Field(
        default="https://paper-api.alpaca.markets", description="Alpaca paper trading base URL"
    )

    live_base_url: str = Field(
        default="https://api.alpaca.markets", description="Alpaca live trading base URL"
    )

    data_feed: str = Field(default="iex", description="Data feed to use (iex or sip)")

    use_polygon: bool = Field(default=False, description="Use Polygon for data instead of Alpaca")

    @model_validator(mode="after")
    def load_credentials(self) -> AlpacaConfig:
        """Load credentials from SecretManager if not set."""
        if not self.api_key_id or not self.api_secret_key:
            secret_manager = get_secret_manager()
            if not self.api_key_id:
                self.api_key_id = secret_manager.get_secret("ALPACA_API_KEY_ID")
            if not self.api_secret_key:
                self.api_secret_key = secret_manager.get_secret("ALPACA_API_SECRET_KEY")
        return self


class OptimizationConfig(BaseModel):
    """Optimization engine configuration."""

    max_workers: int = Field(
        default_factory=lambda: os.cpu_count() or 4,
        description="Maximum number of parallel workers",
    )

    timeout_seconds: int = Field(
        default=300, description="Timeout for optimization runs in seconds", gt=0
    )

    memory_limit_gb: float = Field(
        default=8.0, description="Memory limit for optimization in GB", gt=0
    )

    cache_results: bool = Field(default=True, description="Cache optimization results")

    use_gpu: bool = Field(default=False, description="Use GPU acceleration if available")

    parallel_backtests: int = Field(
        default=4, description="Number of parallel backtests to run", gt=0
    )


class TradingConfig(BaseModel):
    """Unified trading configuration.

    This is the main configuration class that consolidates all settings
    for the GPT-Trader application.
    """

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )

    debug: bool = Field(default=False, description="Enable debug mode")

    # Sub-configurations
    financial: FinancialConfig = Field(
        default_factory=FinancialConfig, description="Financial parameters and constants"
    )

    data: DataConfig = Field(default_factory=DataConfig, description="Data source configuration")

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    alpaca: AlpacaConfig = Field(
        default_factory=AlpacaConfig, description="Alpaca API configuration"
    )

    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig, description="Optimization configuration"
    )

    # Feature flags
    enable_paper_trading: bool = Field(default=True, description="Enable paper trading mode")

    enable_live_trading: bool = Field(
        default=False, description="Enable live trading (use with caution)"
    )

    enable_notifications: bool = Field(default=False, description="Enable trade notifications")

    dry_run: bool = Field(default=False, description="Run in dry-run mode (no actual trades)")

    @model_validator(mode="after")
    def validate_trading_modes(self) -> TradingConfig:
        """Validate trading mode configuration."""
        if self.enable_live_trading and self.environment == Environment.DEVELOPMENT:
            raise ValueError("Cannot enable live trading in development environment")

        if self.enable_live_trading and self.enable_paper_trading:
            raise ValueError("Cannot enable both live and paper trading simultaneously")

        if self.environment == Environment.PRODUCTION and self.debug:
            raise ValueError("Debug mode should not be enabled in production")

        return self

    @classmethod
    def load(cls, config_path: Path | None = None, profile: str | None = None) -> TradingConfig:
        """Load configuration from file or environment.

        Args:
            config_path: Optional path to configuration file.
            profile: Optional configuration profile (dev, test, prod).

        Returns:
            TradingConfig instance.
        """
        # Start with defaults
        config_dict = {}

        # Load from file if provided
        if config_path and config_path.exists():
            import json

            with open(config_path) as f:
                config_dict = json.load(f)

        # Apply profile if specified
        if profile:
            config_dict.update(cls._get_profile_settings(profile))

        # Override from environment variables
        config_dict.update(cls._load_from_env())

        # Create configuration
        return cls(**config_dict)

    @staticmethod
    def _get_profile_settings(profile: str) -> dict[str, Any]:
        """Get settings for a specific profile.

        Args:
            profile: Profile name (dev, test, prod).

        Returns:
            Profile-specific settings.
        """
        profiles = {
            "dev": {
                "environment": Environment.DEVELOPMENT,
                "debug": True,
                "enable_paper_trading": True,
                "enable_live_trading": False,
                "logging": {"level": "DEBUG"},
            },
            "test": {
                "environment": Environment.TESTING,
                "debug": False,
                "enable_paper_trading": True,
                "enable_live_trading": False,
                "dry_run": True,
                "logging": {"level": "INFO"},
            },
            "prod": {
                "environment": Environment.PRODUCTION,
                "debug": False,
                "enable_paper_trading": False,
                "enable_live_trading": True,
                "enable_notifications": True,
                "logging": {"level": "WARNING", "structured_logging": True},
            },
        }

        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}. Must be one of {list(profiles.keys())}")

        return profiles[profile]

    @staticmethod
    def _load_from_env() -> dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration overrides from environment.
        """
        secret_manager = get_secret_manager()
        config = {}

        # Environment
        if env := secret_manager.get_secret("ENVIRONMENT"):
            config["environment"] = Environment(env.lower())

        # Debug mode
        if debug := secret_manager.get_secret("DEBUG"):
            config["debug"] = debug.lower() == "true"

        # Logging level
        if log_level := secret_manager.get_secret("LOG_LEVEL"):
            config.setdefault("logging", {})["level"] = log_level

        # Trading modes
        if paper := secret_manager.get_secret("ENABLE_PAPER_TRADING"):
            config["enable_paper_trading"] = paper.lower() == "true"

        if live := secret_manager.get_secret("ENABLE_LIVE_TRADING"):
            config["enable_live_trading"] = live.lower() == "true"

        return config

    def save(self, path: Path) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration.
        """
        import json
        from decimal import Decimal

        def default_serializer(obj):
            if isinstance(obj, Path | Decimal):
                return str(obj)
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=default_serializer)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def get_initial_capital(self) -> float:
        """Get initial capital based on trading mode.

        Returns:
            Initial capital amount.
        """
        if self.enable_paper_trading:
            return float(self.financial.capital.paper_trading_capital)
        elif self.enable_live_trading:
            return float(self.financial.capital.initial_capital)
        else:
            return float(self.financial.capital.backtesting_capital)


# Singleton instance
_config: TradingConfig | None = None


def get_config() -> TradingConfig:
    """Get the global configuration instance.

    Returns:
        Global TradingConfig instance.
    """
    global _config
    if _config is None:
        _config = TradingConfig.load()
    return _config


def set_config(config: TradingConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: TradingConfig instance to set globally.
    """
    global _config
    _config = config
