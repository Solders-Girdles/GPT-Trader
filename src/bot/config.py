"""Unified Configuration System for GPT-Trader.

This module consolidates all configuration sources into a single, comprehensive
configuration system that eliminates duplicate config modules and provides:

- Type-safe configuration with Pydantic validation
- Environment variable overrides
- YAML/JSON file support
- Financial constants and trading parameters
- Security settings
- Database configuration
- Optimization parameters
- Single get_config() function for the entire codebase
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from bot.security.secrets_manager import get_secret_manager
except ImportError:
    # Fallback for testing or minimal setups
    class MockSecretManager:
        def get_secret(self, key: str, default: str = "") -> str:
            return os.environ.get(key, default)

    def get_secret_manager() -> MockSecretManager:
        return MockSecretManager()


# Initialize secret manager
secret_manager = get_secret_manager()


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityLevel(str, Enum):
    """Security levels."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# Financial Configuration Classes
class TradingLimits(BaseModel):
    """Trading position and risk limits."""

    max_position_size: Decimal = Field(
        default=Decimal("1000000.0"), description="Maximum position size in USD", gt=0
    )
    max_order_value: Decimal = Field(
        default=Decimal("100000.0"), description="Maximum order value in USD", gt=0
    )
    max_portfolio_positions: int = Field(
        default=20, description="Maximum concurrent positions", gt=0
    )
    max_leverage: float = Field(
        default=1.0, description="Maximum leverage multiplier", ge=1.0, le=4.0
    )
    min_position_size: Decimal = Field(
        default=Decimal("100.0"), description="Minimum position size in USD", gt=0
    )
    position_size_increment: Decimal = Field(
        default=Decimal("0.01"), description="Minimum position sizing increment", gt=0
    )


class RiskParameters(BaseModel):
    """Risk management parameters."""

    max_portfolio_risk: float = Field(
        default=0.02, description="Maximum portfolio risk per trade (2%)", gt=0, le=0.1
    )
    max_daily_loss: Decimal = Field(
        default=Decimal("5000.0"), description="Maximum daily loss limit in USD", gt=0
    )
    max_drawdown_percent: float = Field(
        default=0.20, description="Maximum acceptable drawdown (20%)", gt=0, le=0.5
    )
    stop_loss_percent: float = Field(
        default=0.02, description="Default stop loss percentage (2%)", gt=0, le=0.1
    )
    take_profit_percent: float = Field(
        default=0.05, description="Default take profit percentage (5%)", gt=0, le=0.5
    )
    risk_free_rate: float = Field(
        default=0.05, description="Annual risk-free rate for Sharpe ratio", ge=0, le=0.2
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level for VaR calculations", gt=0.9, lt=1.0
    )


class TransactionCosts(BaseModel):
    """Transaction cost configuration."""

    commission_per_share: Decimal = Field(
        default=Decimal("0.005"), description="Commission per share in USD", ge=0
    )
    commission_minimum: Decimal = Field(
        default=Decimal("1.0"), description="Minimum commission per trade in USD", ge=0
    )
    commission_rate_bps: float = Field(
        default=10.0, description="Commission rate in basis points", ge=0, le=100
    )
    slippage_bps: float = Field(
        default=5.0, description="Expected slippage in basis points", ge=0, le=100
    )
    market_impact_bps: float = Field(
        default=2.0, description="Expected market impact in basis points", ge=0, le=100
    )

    @property
    def commission_rate_decimal(self) -> float:
        """Convert commission rate from basis points to decimal."""
        return self.commission_rate_bps / 10000.0

    @property
    def slippage_decimal(self) -> float:
        """Convert slippage from basis points to decimal."""
        return self.slippage_bps / 10000.0

    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points."""
        return self.commission_rate_bps + self.slippage_bps + self.market_impact_bps


class CapitalAllocation(BaseModel):
    """Capital allocation configuration."""

    initial_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Initial trading capital in USD", gt=0
    )
    paper_trading_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Paper trading capital in USD", gt=0
    )
    backtesting_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Backtesting capital in USD", gt=0
    )
    deployment_budget: Decimal = Field(
        default=Decimal("10000.0"), description="Strategy deployment budget in USD", gt=0
    )
    reserve_capital_percent: float = Field(
        default=0.20, description="Percentage of capital in reserve (20%)", ge=0, le=0.5
    )
    max_capital_per_strategy: Decimal = Field(
        default=Decimal("50000.0"), description="Maximum capital per strategy in USD", gt=0
    )
    min_capital_per_strategy: Decimal = Field(
        default=Decimal("1000.0"), description="Minimum capital per strategy in USD", gt=0
    )

    @field_validator("initial_capital", "paper_trading_capital", "backtesting_capital")
    @classmethod
    def validate_capital_amounts(cls, v: Decimal) -> Decimal:
        """Validate capital amounts are reasonable."""
        if v < 1000:
            raise ValueError(f"Capital must be at least $1,000, got ${v}")
        if v > 100_000_000:
            raise ValueError(f"Capital exceeds maximum limit of $100M, got ${v}")
        return v

    @property
    def available_trading_capital(self) -> Decimal:
        """Calculate available trading capital after reserves."""
        return self.initial_capital * Decimal(str(1 - self.reserve_capital_percent))


class FinancialConfig(BaseModel):
    """Comprehensive financial configuration."""

    capital: CapitalAllocation = Field(default_factory=CapitalAllocation)
    limits: TradingLimits = Field(default_factory=TradingLimits)
    risk: RiskParameters = Field(default_factory=RiskParameters)
    costs: TransactionCosts = Field(default_factory=TransactionCosts)
    base_currency: str = Field(default="USD", description="Base currency for calculations")
    decimal_precision: int = Field(
        default=2, description="Decimal precision for calculations", ge=0, le=8
    )
    market_open_hour: int = Field(
        default=9, description="Market open hour (24-hour format)", ge=0, lt=24
    )
    market_open_minute: int = Field(default=30, description="Market open minute", ge=0, lt=60)
    market_close_hour: int = Field(
        default=16, description="Market close hour (24-hour format)", ge=0, lt=24
    )
    market_close_minute: int = Field(default=0, description="Market close minute", ge=0, lt=60)
    max_historical_days: int = Field(
        default=730, description="Maximum days of historical data (2 years)", gt=30
    )


# Data Configuration
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
    default_source: str = Field(default="yfinance", description="Default data source")
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


# Logging Configuration
class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
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

    @property
    def max_size_bytes(self) -> int:
        """Get max size in bytes."""
        return self.max_size_mb * 1024 * 1024


# Alpaca Configuration
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
            if not self.api_key_id:
                self.api_key_id = secret_manager.get_secret("ALPACA_API_KEY_ID")
            if not self.api_secret_key:
                self.api_secret_key = secret_manager.get_secret("ALPACA_API_SECRET_KEY")
        return self


# Database Configuration
class DatabaseConfig(BaseModel):
    """Database configuration."""

    database_path: Path = Field(
        default=Path("data/gpt_trader.db"), description="Database file path"
    )
    timeout: float = Field(default=30.0, description="Database timeout in seconds", gt=0)
    max_connections: int = Field(default=20, description="Maximum database connections", gt=0)
    journal_mode: str = Field(default="WAL", description="SQLite journal mode")
    synchronous: str = Field(default="NORMAL", description="SQLite synchronous mode")
    cache_size: int = Field(default=-64000, description="Cache size (-64MB)")  # 64MB
    backup_enabled: bool = Field(default=True, description="Enable database backups")
    backup_interval_hours: int = Field(default=6, description="Backup interval in hours", gt=0)
    backup_retention_days: int = Field(default=7, description="Backup retention in days", gt=0)


# Security Configuration
class SecurityConfig(BaseModel):
    """Security configuration."""

    security_level: SecurityLevel = Field(
        default=SecurityLevel.PRODUCTION, description="Security level"
    )
    debug_mode: bool = Field(default=False, description="Debug mode")
    jwt_secret_key: str = Field(default="", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT expiry in hours", gt=0)
    api_rate_limit_per_minute: int = Field(
        default=60, description="API rate limit per minute", gt=0
    )
    require_https: bool = Field(default=True, description="Require HTTPS")
    encrypt_data_at_rest: bool = Field(default=True, description="Encrypt data at rest")
    enable_security_headers: bool = Field(default=True, description="Enable security headers")

    @model_validator(mode="after")
    def validate_security_settings(self) -> SecurityConfig:
        """Validate security settings based on level."""
        if self.security_level == SecurityLevel.DEVELOPMENT:
            self.debug_mode = True
            self.require_https = False
            self.api_rate_limit_per_minute = 1000
        elif self.security_level == SecurityLevel.PRODUCTION:
            self.debug_mode = False
            self.require_https = True
            self.encrypt_data_at_rest = True
            if not self.jwt_secret_key:
                self.jwt_secret_key = secret_manager.get_secret("JWT_SECRET_KEY", "")
        return self


# Optimization Configuration
class OptimizationConfig(BaseModel):
    """Optimization engine configuration."""

    max_workers: int = Field(
        default_factory=lambda: os.cpu_count() or 4, description="Maximum parallel workers"
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


# Main Configuration Class
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
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
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
            with open(config_path) as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f) or {}
                else:
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
        """Get settings for a specific profile."""
        profiles = {
            "dev": {
                "environment": Environment.DEVELOPMENT,
                "debug": True,
                "enable_paper_trading": True,
                "enable_live_trading": False,
                "logging": {"level": LogLevel.DEBUG},
                "security": {"security_level": SecurityLevel.DEVELOPMENT},
            },
            "test": {
                "environment": Environment.TESTING,
                "debug": False,
                "enable_paper_trading": True,
                "enable_live_trading": False,
                "dry_run": True,
                "logging": {"level": LogLevel.INFO},
                "security": {"security_level": SecurityLevel.STAGING},
            },
            "prod": {
                "environment": Environment.PRODUCTION,
                "debug": False,
                "enable_paper_trading": False,
                "enable_live_trading": True,
                "enable_notifications": True,
                "logging": {"level": LogLevel.WARNING, "structured_logging": True},
                "security": {"security_level": SecurityLevel.PRODUCTION},
            },
        }

        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}. Must be one of {list(profiles.keys())}")

        return profiles[profile]

    @staticmethod
    def _load_from_env() -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Environment
        if env := secret_manager.get_secret("ENVIRONMENT"):
            try:
                config["environment"] = Environment(env.lower())
            except ValueError:
                pass  # Keep default

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
        """Save configuration to file."""

        def default_serializer(obj: Any) -> str:
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
        """Get initial capital based on trading mode."""
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

    This is the single source of truth for configuration across the entire codebase.
    All other config modules and functions should be replaced with this one.

    Returns:
        Global TradingConfig instance.
    """
    global _config
    if _config is None:
        _config = TradingConfig.load()
        # Ensure cache directory exists
        _config.data.cache_dir.mkdir(parents=True, exist_ok=True)
        # Ensure database directory exists
        _config.database.database_path.parent.mkdir(parents=True, exist_ok=True)
    return _config


def set_config(config: TradingConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: TradingConfig instance to set globally.
    """
    global _config
    _config = config


def reload_config() -> TradingConfig:
    """Reload configuration from environment."""
    global _config
    _config = None
    return get_config()


# Legacy support - these will be deprecated
settings = None  # Will be set when get_config() is first called


def __getattr__(name: str) -> Any:
    """Provide backward compatibility for legacy imports."""
    if name == "settings":
        global settings
        if settings is None:
            settings = get_config()
        return settings
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Configuration utilities
class ConfigManager:
    """Configuration management utilities."""

    @staticmethod
    def load_yaml_config(path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def save_yaml_config(config: dict[str, Any], path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(config, f, indent=2, default_flow_style=False)

    @staticmethod
    def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
