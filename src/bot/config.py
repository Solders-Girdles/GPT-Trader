from __future__ import annotations

import os
from pathlib import Path

from bot.security.secrets_manager import get_secret_manager
from bot.utils.base import BaseConfig
from pydantic import BaseModel, Field, validator

# Initialize secret manager instead of using dotenv
secret_manager = get_secret_manager()


class AlpacaSettings(BaseModel):
    """Alpaca trading API configuration."""

    api_key_id: str | None = Field(
        default_factory=lambda: secret_manager.get_secret("ALPACA_API_KEY_ID"),
        description="Alpaca API key ID",
    )
    api_secret_key: str | None = Field(
        default_factory=lambda: secret_manager.get_secret("ALPACA_API_SECRET_KEY"),
        description="Alpaca API secret key",
    )
    paper_base_url: str = Field(
        default="https://paper-api.alpaca.markets", description="Alpaca paper trading base URL"
    )
    live_base_url: str = Field(
        default="https://api.alpaca.markets", description="Alpaca live trading base URL"
    )


class DataSettings(BaseModel):
    """Data source configuration."""

    cache_dir: Path = Field(
        default=Path("data/cache"), description="Directory for caching market data"
    )
    strict_validation: bool = Field(
        default=True, description="Whether to strictly validate OHLC data"
    )
    default_source: str = Field(
        default="yfinance", description="Default data source (yfinance, alpaca, etc.)"
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default_factory=lambda: secret_manager.get_secret("LOG_LEVEL", "INFO"),
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: Path | None = Field(
        default=None, description="Log file path (if None, logs to console only)"
    )
    max_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum log file size in bytes"  # 10MB
    )
    backup_count: int = Field(default=5, description="Number of backup log files to keep")


class OptimizationSettings(BaseModel):
    """Optimization configuration."""

    max_workers: int = Field(
        default=os.cpu_count() or 4, description="Maximum number of parallel workers"
    )
    timeout_seconds: int = Field(
        default=300, description="Timeout for optimization runs in seconds"
    )
    memory_limit_gb: float = Field(default=8.0, description="Memory limit for optimization in GB")


class Settings(BaseConfig):
    """Main application settings."""

    log_level: str = Field(default="INFO", description="Logging level")
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)

    # Environment-specific settings
    environment: str = Field(
        default=secret_manager.get_secret("ENVIRONMENT", "development"),
        description="Environment (development, testing, production)",
    )
    debug: bool = Field(
        default=secret_manager.get_secret("DEBUG", "false").lower() == "true",
        description="Debug mode",
    )

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = {"development", "testing", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure cache directory exists
settings.data.cache_dir.mkdir(parents=True, exist_ok=True)
