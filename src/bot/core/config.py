"""
GPT-Trader Centralized Configuration Management

Unified configuration system replacing hard-coded values throughout the codebase:

- Environment-specific configuration (dev/staging/prod)
- Type-safe configuration with validation
- Runtime configuration updates
- Secure secrets management
- Configuration inheritance and overrides
- Default value management

Supports multiple configuration sources:
- YAML/TOML configuration files
- Environment variables
- Command-line arguments
- Database configuration store
- Runtime API updates
"""

import logging
import os
from dataclasses import dataclass, field, fields
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import toml
import yaml

from .exceptions import (
    raise_config_error,
    raise_validation_error,
)

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigSource(Enum):
    """Configuration sources"""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    COMMAND_LINE = "command_line"
    DATABASE = "database"
    API = "api"


@dataclass
class DatabaseConfig:
    """Database configuration"""

    database_path: Path = field(default_factory=lambda: Path("data/gpt_trader.db"))
    timeout: float = 30.0
    max_connections: int = 20
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000  # 64MB
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 7

    def __post_init__(self):
        """Validate database configuration"""
        if self.timeout <= 0:
            raise_validation_error("Database timeout must be positive", "timeout", self.timeout)
        if self.max_connections <= 0:
            raise_validation_error(
                "Max connections must be positive", "max_connections", self.max_connections
            )
        if isinstance(self.database_path, str):
            self.database_path = Path(self.database_path)


@dataclass
class TradingConfig:
    """Trading engine configuration"""

    initial_capital: Decimal = Decimal("100000.0")
    enable_live_trading: bool = False
    max_order_value_pct: float = 0.10  # 10% of capital
    max_position_pct: float = 0.20  # 20% of capital
    max_daily_trades: int = 100
    order_timeout_seconds: int = 30
    execution_thread_count: int = 3

    # Broker settings
    broker_type: str = "alpaca_paper"
    broker_api_key: str = ""
    broker_secret_key: str = ""
    broker_base_url: str = "https://paper-api.alpaca.markets"

    # Risk settings
    daily_loss_limit_pct: float = 0.05  # 5% daily loss limit
    max_drawdown_pct: float = 0.15  # 15% maximum drawdown
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

    def __post_init__(self):
        """Validate trading configuration"""
        if self.initial_capital <= 0:
            raise_validation_error(
                "Initial capital must be positive", "initial_capital", self.initial_capital
            )
        if not (0 < self.max_order_value_pct <= 1):
            raise_validation_error(
                "Max order value percentage must be between 0 and 1",
                "max_order_value_pct",
                self.max_order_value_pct,
            )
        if not (0 < self.max_position_pct <= 1):
            raise_validation_error(
                "Max position percentage must be between 0 and 1",
                "max_position_pct",
                self.max_position_pct,
            )


@dataclass
class RiskConfig:
    """Risk management configuration"""

    enable_risk_monitoring: bool = True
    risk_calculation_interval_seconds: int = 30
    position_update_interval_seconds: int = 10

    # VaR/CVaR settings
    var_confidence_level: float = 0.95
    var_time_horizon_days: int = 1
    historical_window_days: int = 252  # 1 year

    # Circuit breaker settings
    enable_circuit_breakers: bool = True
    enable_auto_liquidation: bool = True
    circuit_breaker_cooldown_minutes: int = 15
    max_triggers_per_day: int = 3

    # Drawdown limits
    daily_loss_threshold_pct: float = 0.05  # 5%
    portfolio_drawdown_threshold_pct: float = 0.15  # 15%
    position_concentration_threshold_pct: float = 0.20  # 20%
    volatility_spike_threshold: float = 3.0  # 3x normal volatility

    def __post_init__(self):
        """Validate risk configuration"""
        if not (0 < self.var_confidence_level < 1):
            raise_validation_error(
                "VaR confidence level must be between 0 and 1",
                "var_confidence_level",
                self.var_confidence_level,
            )
        if self.historical_window_days <= 0:
            raise_validation_error(
                "Historical window must be positive",
                "historical_window_days",
                self.historical_window_days,
            )


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""

    enable_monitoring: bool = True
    dashboard_update_interval_seconds: float = 1.0
    health_check_interval_seconds: int = 30
    metrics_collection_enabled: bool = True

    # Alerting settings
    enable_alerting: bool = True
    alert_worker_threads: int = 3
    alert_queue_size: int = 1000

    # Channel configurations
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from_address: str = ""
    email_to_addresses: list[str] = field(default_factory=list)

    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#gpt-trader-alerts"

    sms_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_phone: str = ""
    twilio_to_phones: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Market data configuration"""

    primary_data_source: str = "yfinance"
    backup_data_sources: list[str] = field(default_factory=lambda: ["polygon", "alpaca"])

    # Streaming settings
    enable_streaming: bool = True
    streaming_update_interval_seconds: float = 0.1  # 100ms
    bar_intervals: list[str] = field(default_factory=lambda: ["1min", "5min", "15min", "1hour"])

    # Data quality settings
    max_quote_spread_pct: float = 0.10  # 10% max bid-ask spread
    max_price_change_pct: float = 0.20  # 20% max price change
    stale_data_threshold_seconds: int = 30

    # API settings
    yfinance_rate_limit: int = 100  # requests per minute
    polygon_api_key: str = ""
    alpaca_data_api_key: str = ""
    iex_api_key: str = ""

    # Storage settings
    quote_sampling_rate: float = 0.01  # Store 1% of quotes
    trade_storage_enabled: bool = True
    bar_storage_enabled: bool = True
    data_retention_days: int = 365


@dataclass
class StrategyConfig:
    """Strategy development and health monitoring configuration"""

    enable_strategy_health_monitoring: bool = True
    health_check_interval_minutes: int = 15
    min_trades_for_analysis: int = 10

    # Performance thresholds
    min_win_rate: float = 0.30  # 30% minimum win rate
    min_profit_factor: float = 1.2  # 1.2 minimum profit factor
    max_drawdown_threshold: float = 0.20  # 20% maximum drawdown
    min_sharpe_ratio: float = 0.5  # 0.5 minimum Sharpe ratio

    # Strategy lifecycle
    auto_disable_failed_strategies: bool = True
    auto_liquidate_on_failure: bool = False  # Safer default
    strategy_failure_threshold_hours: int = 24

    # Optimization settings
    enable_strategy_optimization: bool = True
    optimization_lookback_days: int = 30
    parameter_optimization_enabled: bool = False


@dataclass
class SystemConfig:
    """Overall system configuration"""

    # Environment and deployment
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = True
    log_level: LogLevel = LogLevel.INFO

    # Directory structure
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    config_dir: Path = field(default_factory=lambda: Path("config"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # System resource limits
    max_memory_usage_gb: float = 4.0
    max_cpu_usage_pct: float = 80.0
    max_disk_usage_pct: float = 85.0

    def __post_init__(self):
        """Validate and normalize system configuration"""
        # Convert string paths to Path objects
        for field_name in ["data_dir", "logs_dir", "config_dir", "cache_dir"]:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

        # Validate resource limits
        if self.max_memory_usage_gb <= 0:
            raise_validation_error(
                "Max memory usage must be positive", "max_memory_usage_gb", self.max_memory_usage_gb
            )

        # Ensure directories are absolute for production
        if self.environment == Environment.PRODUCTION:
            for field_name in ["data_dir", "logs_dir", "config_dir", "cache_dir"]:
                path = getattr(self, field_name)
                if not path.is_absolute():
                    raise_config_error(f"{field_name} must be absolute path in production")

    def create_directories(self) -> None:
        """Create necessary directories"""
        directories = [self.data_dir, self.logs_dir, self.config_dir, self.cache_dir]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise_config_error(f"Failed to create directory {directory}: {str(e)}")


class ConfigurationManager:
    """
    Centralized configuration management system

    Handles loading, validation, and management of all system configuration
    with support for multiple sources and runtime updates.
    """

    def __init__(
        self, config_file: Path | None = None, environment: Environment | None = None
    ) -> None:
        """Initialize configuration manager"""
        self.config_file = config_file
        self.environment = environment or Environment.DEVELOPMENT
        self.config_sources: dict[str, ConfigSource] = {}
        self.config: SystemConfig | None = None
        self._secrets: dict[str, str] = {}

        # Load configuration
        self._load_configuration()

        logger.info(f"Configuration manager initialized for environment: {self.environment.value}")

    def _load_configuration(self) -> None:
        """Load configuration from all sources"""
        config_data = {}

        # 1. Load defaults
        self.config = SystemConfig()
        config_data.update(self._config_to_dict(self.config))
        self.config_sources.update({k: ConfigSource.DEFAULT for k in config_data.keys()})

        # 2. Load from file if specified
        if self.config_file and self.config_file.exists():
            file_data = self._load_config_file(self.config_file)
            config_data.update(file_data)
            self.config_sources.update({k: ConfigSource.FILE for k in file_data.keys()})

        # 3. Load from environment variables
        env_data = self._load_environment_variables()
        config_data.update(env_data)
        self.config_sources.update({k: ConfigSource.ENVIRONMENT for k in env_data.keys()})

        # 4. Load secrets
        self._load_secrets()

        # 5. Build final configuration
        self.config = self._build_config_from_data(config_data)

        # 6. Validate configuration
        self._validate_configuration()

        # 7. Create directories
        self.config.create_directories()

    def _load_config_file(self, config_file: Path) -> dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_file) as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == ".toml":
                    return toml.load(f) or {}
                else:
                    raise_config_error(f"Unsupported config file format: {config_file.suffix}")
        except Exception as e:
            raise_config_error(f"Failed to load config file {config_file}: {str(e)}")

    def _load_environment_variables(self) -> dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        prefix = "GPT_TRADER_"

        # Map environment variables to config keys
        env_mappings = {
            f"{prefix}ENVIRONMENT": "environment",
            f"{prefix}DEBUG": "debug_mode",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}DATA_DIR": "data_dir",
            f"{prefix}LOGS_DIR": "logs_dir",
            # Database
            f"{prefix}DB_PATH": "database.database_path",
            f"{prefix}DB_TIMEOUT": "database.timeout",
            f"{prefix}DB_MAX_CONNECTIONS": "database.max_connections",
            # Trading
            f"{prefix}INITIAL_CAPITAL": "trading.initial_capital",
            f"{prefix}LIVE_TRADING": "trading.enable_live_trading",
            f"{prefix}BROKER_TYPE": "trading.broker_type",
            f"{prefix}BROKER_API_KEY": "trading.broker_api_key",
            f"{prefix}BROKER_SECRET_KEY": "trading.broker_secret_key",
            # Risk
            f"{prefix}DAILY_LOSS_LIMIT": "risk.daily_loss_threshold_pct",
            f"{prefix}MAX_DRAWDOWN": "risk.portfolio_drawdown_threshold_pct",
            f"{prefix}ENABLE_CIRCUIT_BREAKERS": "risk.enable_circuit_breakers",
            # Monitoring
            f"{prefix}EMAIL_ENABLED": "monitoring.email_enabled",
            f"{prefix}EMAIL_USERNAME": "monitoring.email_username",
            f"{prefix}EMAIL_PASSWORD": "monitoring.email_password",
            f"{prefix}SLACK_WEBHOOK": "monitoring.slack_webhook_url",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                env_config[config_key] = self._convert_env_value(value)

        return env_config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _load_secrets(self) -> None:
        """Load secrets from secure sources"""
        # In production, this would integrate with:
        # - AWS Secrets Manager
        # - Azure Key Vault
        # - HashiCorp Vault
        # - Kubernetes Secrets

        # For now, load from environment variables with _SECRET suffix
        secret_prefix = "GPT_TRADER_SECRET_"

        for env_var, value in os.environ.items():
            if env_var.startswith(secret_prefix):
                secret_name = env_var[len(secret_prefix) :].lower()
                self._secrets[secret_name] = value

    def _config_to_dict(self, config: SystemConfig) -> dict[str, Any]:
        """Convert configuration object to dictionary"""
        result = {}

        for field in fields(config):
            value = getattr(config, field.name)

            if hasattr(value, "__dataclass_fields__"):
                # Nested dataclass
                nested_dict = {}
                for nested_field in fields(value):
                    nested_value = getattr(value, nested_field.name)
                    nested_dict[nested_field.name] = nested_value
                result[field.name] = nested_dict
            else:
                result[field.name] = value

        return result

    def _build_config_from_data(self, config_data: dict[str, Any]) -> SystemConfig:
        """Build configuration object from data dictionary"""
        # This is a simplified implementation
        # In production, would use proper deserialization with type conversion

        # Extract nested configurations
        database_data = config_data.get("database", {})
        trading_data = config_data.get("trading", {})
        risk_data = config_data.get("risk", {})
        monitoring_data = config_data.get("monitoring", {})
        data_data = config_data.get("data", {})
        strategy_data = config_data.get("strategy", {})

        # Build nested configs
        database_config = DatabaseConfig(
            **{
                k: v
                for k, v in database_data.items()
                if k in [f.name for f in fields(DatabaseConfig)]
            }
        )
        trading_config = TradingConfig(
            **{
                k: v
                for k, v in trading_data.items()
                if k in [f.name for f in fields(TradingConfig)]
            }
        )
        risk_config = RiskConfig(
            **{k: v for k, v in risk_data.items() if k in [f.name for f in fields(RiskConfig)]}
        )
        monitoring_config = MonitoringConfig(
            **{
                k: v
                for k, v in monitoring_data.items()
                if k in [f.name for f in fields(MonitoringConfig)]
            }
        )
        data_config = DataConfig(
            **{k: v for k, v in data_data.items() if k in [f.name for f in fields(DataConfig)]}
        )
        strategy_config = StrategyConfig(
            **{
                k: v
                for k, v in strategy_data.items()
                if k in [f.name for f in fields(StrategyConfig)]
            }
        )

        # Build main config
        main_config_data = {
            k: v
            for k, v in config_data.items()
            if k not in ["database", "trading", "risk", "monitoring", "data", "strategy"]
        }

        return SystemConfig(
            database=database_config,
            trading=trading_config,
            risk=risk_config,
            monitoring=monitoring_config,
            data=data_config,
            strategy=strategy_config,
            **{
                k: v
                for k, v in main_config_data.items()
                if k in [f.name for f in fields(SystemConfig)]
            },
        )

    def _validate_configuration(self) -> None:
        """Validate complete configuration"""
        if not self.config:
            raise_config_error("Configuration is None")

        # Environment-specific validations
        if self.config.environment == Environment.PRODUCTION:
            self._validate_production_config()

        # Cross-component validations
        self._validate_cross_component_config()

    def _validate_production_config(self) -> None:
        """Validate production-specific requirements"""
        # Ensure sensitive data is not in debug mode
        if self.config.debug_mode:
            logger.warning("Debug mode enabled in production environment")

        # Ensure proper logging level
        if self.config.log_level in [LogLevel.DEBUG]:
            logger.warning("Debug logging enabled in production")

        # Validate broker credentials for live trading
        if self.config.trading.enable_live_trading:
            if not self.config.trading.broker_api_key:
                raise_config_error("Broker API key required for live trading")
            if not self.config.trading.broker_secret_key:
                raise_config_error("Broker secret key required for live trading")

        # Validate alerting configuration
        if self.config.monitoring.enable_alerting:
            has_alert_channel = (
                self.config.monitoring.email_enabled
                or self.config.monitoring.slack_enabled
                or self.config.monitoring.sms_enabled
            )
            if not has_alert_channel:
                raise_config_error("At least one alert channel must be enabled")

    def _validate_cross_component_config(self) -> None:
        """Validate cross-component configuration consistency"""
        # Ensure risk limits are consistent
        if (
            self.config.risk.daily_loss_threshold_pct
            > self.config.risk.portfolio_drawdown_threshold_pct
        ):
            logger.warning("Daily loss threshold exceeds portfolio drawdown threshold")

        # Ensure monitoring intervals are reasonable
        if self.config.monitoring.dashboard_update_interval_seconds < 0.1:
            logger.warning("Dashboard update interval very frequent, may impact performance")

        # Validate data retention vs disk limits
        # This would check estimated data volumes against disk limits

    def get_config(self) -> SystemConfig:
        """Get current system configuration"""
        if self.config is None:
            raise_config_error("Configuration not loaded")
        return self.config

    def get_secret(self, secret_name: str) -> str | None:
        """Get a secret value"""
        return self._secrets.get(secret_name.lower())

    def update_config(
        self, config_path: str, value: Any, source: ConfigSource = ConfigSource.API
    ) -> None:
        """Update configuration value at runtime"""
        # This would implement runtime configuration updates
        # For now, just log the change
        logger.info(
            f"Configuration update requested: {config_path} = {value} (source: {source.value})"
        )

    def save_config(self, file_path: Path | None = None) -> None:
        """Save current configuration to file"""
        if file_path is None:
            file_path = self.config.config_dir / "current_config.yaml"

        config_dict = self._config_to_dict(self.config)

        try:
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            raise_config_error(f"Failed to save configuration: {str(e)}")

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            "environment": self.config.environment.value,
            "debug_mode": self.config.debug_mode,
            "log_level": self.config.log_level.value,
            "live_trading_enabled": self.config.trading.enable_live_trading,
            "risk_monitoring_enabled": self.config.risk.enable_risk_monitoring,
            "alerting_enabled": self.config.monitoring.enable_alerting,
            "data_streaming_enabled": self.config.data.enable_streaming,
            "config_sources": len(self.config_sources),
            "secrets_loaded": len(self._secrets),
        }


# Global configuration manager
_config_manager: ConfigurationManager | None = None


def initialize_config(
    config_file: Path | None = None, environment: Environment | None = None
) -> ConfigurationManager:
    """Initialize global configuration manager"""
    global _config_manager
    _config_manager = ConfigurationManager(config_file, environment)
    return _config_manager


def get_config() -> SystemConfig:
    """Get global system configuration"""
    if _config_manager is None:
        raise_config_error("Configuration not initialized. Call initialize_config() first.")
    return _config_manager.get_config()


def get_secret(secret_name: str) -> str | None:
    """Get a secret value"""
    if _config_manager is None:
        raise_config_error("Configuration not initialized. Call initialize_config() first.")
    return _config_manager.get_secret(secret_name)


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager"""
    if _config_manager is None:
        raise_config_error("Configuration not initialized. Call initialize_config() first.")
    return _config_manager
