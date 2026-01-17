"""
Unified profile loading from YAML configuration files.

Provides YAML-first profile configuration with hardcoded fallbacks.
All profiles can be defined in YAML files under config/profiles/.

Schema Structure:
    profile_name: str
    environment: str
    trading:
        symbols: list[str]
        mode: str  # normal | reduce_only | paper
    strategy:
        type: str
        short_ma_period: int
        long_ma_period: int
    risk_management:
        max_leverage: int
        max_position_size: Decimal
        enable_shorts: bool
        daily_loss_limit_pct: float
    execution:
        time_in_force: str
        dry_run: bool
        mock_broker: bool
    session:
        start_time: str (HH:MM)
        end_time: str (HH:MM)
        trading_days: list[str]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.config import path_registry
from gpt_trader.config.types import Profile
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.config.bot_config import BotConfig

logger = get_logger(__name__, component="profile_loader")


@dataclass
class TradingConfig:
    """Trading section of profile configuration."""

    symbols: list[str] = field(default_factory=lambda: ["BTC-USD"])
    mode: str = "normal"  # normal | reduce_only | paper
    interval: int = 60  # seconds


@dataclass
class StrategyConfig:
    """Strategy section of profile configuration."""

    type: str = "baseline"  # baseline | mean_reversion
    short_ma_period: int = 5
    long_ma_period: int = 20
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


@dataclass
class RiskConfig:
    """Risk management section of profile configuration."""

    max_leverage: int = 1
    max_position_size: Decimal = field(default_factory=lambda: Decimal("10000"))
    position_fraction: Decimal = field(default_factory=lambda: Decimal("0.1"))
    enable_shorts: bool = False
    daily_loss_limit_pct: float = 0.05  # Percentage of equity (0.05 = 5%)
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))


@dataclass
class ExecutionConfig:
    """Execution section of profile configuration."""

    time_in_force: str = "GTC"  # GTC | IOC | FOK
    dry_run: bool = False
    mock_broker: bool = False
    mock_fills: bool = False
    use_limit_orders: bool = False
    market_order_fallback: bool = True


@dataclass
class SessionConfig:
    """Session timing section of profile configuration."""

    start_time: time | None = None
    end_time: time | None = None
    trading_days: list[str] = field(
        default_factory=lambda: ["monday", "tuesday", "wednesday", "thursday", "friday"]
    )


@dataclass
class MonitoringConfig:
    """Monitoring section of profile configuration."""

    log_level: str = "INFO"
    update_interval: int = 60  # seconds
    status_enabled: bool = True


@dataclass
class ProfileSchema:
    """Complete profile schema with all configuration sections.

    This dataclass represents a unified profile configuration that can be:
    - Loaded from YAML files
    - Used as hardcoded defaults
    - Merged with environment overrides
    """

    profile_name: str
    environment: str = "development"
    description: str = ""

    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, data: dict[str, Any], profile_name: str) -> ProfileSchema:
        """Create ProfileSchema from parsed YAML data."""
        trading_data = data.get("trading", {})
        strategy_data = data.get("strategy", {})
        risk_data = data.get("risk_management", {})
        execution_data = data.get("execution", {})
        if not execution_data:
            execution_data = trading_data.get("execution", {})
        session_data = data.get("session", {})
        monitoring_data = data.get("monitoring", {})

        # Parse trading config
        trading = TradingConfig(
            symbols=trading_data.get("symbols", ["BTC-USD"]),
            mode=trading_data.get("mode", "normal"),
            interval=int(trading_data.get("interval", 60)),
        )

        # Parse strategy config
        strategy = StrategyConfig(
            type=strategy_data.get("type", "baseline"),
            short_ma_period=int(strategy_data.get("short_ma_period", 5)),
            long_ma_period=int(strategy_data.get("long_ma_period", 20)),
            rsi_period=int(strategy_data.get("rsi_period", 14)),
            rsi_overbought=int(strategy_data.get("rsi_overbought", 70)),
            rsi_oversold=int(strategy_data.get("rsi_oversold", 30)),
        )

        # Parse risk config
        risk = RiskConfig(
            max_leverage=int(risk_data.get("max_leverage", 1)),
            max_position_size=Decimal(str(risk_data.get("max_position_size", 10000))),
            position_fraction=Decimal(str(risk_data.get("position_fraction", "0.1"))),
            enable_shorts=risk_data.get("enable_shorts", False),
            daily_loss_limit_pct=float(risk_data.get("daily_loss_limit_pct", 0.05)),
            stop_loss_pct=Decimal(str(risk_data.get("stop_loss_pct", "0.02"))),
            take_profit_pct=Decimal(str(risk_data.get("take_profit_pct", "0.04"))),
        )

        # Parse execution config
        execution = ExecutionConfig(
            time_in_force=execution_data.get("time_in_force", "GTC"),
            dry_run=execution_data.get("dry_run", False),
            mock_broker=execution_data.get("mock_broker", False),
            mock_fills=execution_data.get("mock_fills", False),
            use_limit_orders=execution_data.get("use_limit_orders", False),
            market_order_fallback=execution_data.get("market_order_fallback", True),
        )

        # Parse session config
        start_time = None
        end_time = None
        if "start_time" in session_data:
            try:
                start_time = time.fromisoformat(session_data["start_time"])
            except (ValueError, TypeError):
                pass
        if "end_time" in session_data:
            try:
                end_time = time.fromisoformat(session_data["end_time"])
            except (ValueError, TypeError):
                pass

        session = SessionConfig(
            start_time=start_time,
            end_time=end_time,
            trading_days=session_data.get(
                "trading_days",
                ["monday", "tuesday", "wednesday", "thursday", "friday"],
            ),
        )

        # Parse monitoring config
        metrics_data = monitoring_data.get("metrics", {})
        monitoring = MonitoringConfig(
            log_level=monitoring_data.get("log_level", "INFO"),
            update_interval=int(metrics_data.get("interval_seconds", 60)),
            status_enabled=monitoring_data.get("status_enabled", True),
        )

        return cls(
            profile_name=data.get("profile_name", profile_name),
            environment=data.get("environment", "development"),
            description=data.get("description", ""),
            trading=trading,
            strategy=strategy,
            risk=risk,
            execution=execution,
            session=session,
            monitoring=monitoring,
        )


# Hardcoded profile defaults for when YAML files don't exist
_PROFILE_DEFAULTS: dict[Profile, ProfileSchema] = {
    Profile.DEV: ProfileSchema(
        profile_name="dev",
        environment="development",
        description="Development profile with mock broker",
        trading=TradingConfig(symbols=["BTC-USD", "ETH-USD"], mode="normal"),
        risk=RiskConfig(max_position_size=Decimal("10000"), enable_shorts=True),
        execution=ExecutionConfig(
            dry_run=True,
            mock_broker=True,
            mock_fills=True,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
        monitoring=MonitoringConfig(log_level="DEBUG"),
    ),
    Profile.TEST: ProfileSchema(
        profile_name="test",
        environment="test",
        description="Test profile for unit/integration tests",
        trading=TradingConfig(symbols=["BTC-USD"], mode="normal"),
        risk=RiskConfig(max_position_size=Decimal("1000")),
        execution=ExecutionConfig(
            dry_run=True,
            mock_broker=True,
            mock_fills=True,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
        monitoring=MonitoringConfig(log_level="DEBUG"),
    ),
    Profile.DEMO: ProfileSchema(
        profile_name="demo",
        environment="demo",
        description="Conservative demo profile for demonstrations",
        trading=TradingConfig(symbols=["BTC-USD"], mode="normal"),
        risk=RiskConfig(
            max_position_size=Decimal("100"),
            max_leverage=1,
            enable_shorts=False,
        ),
        execution=ExecutionConfig(
            dry_run=False,
            mock_broker=False,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
    ),
    Profile.SPOT: ProfileSchema(
        profile_name="spot",
        environment="production",
        description="Production spot trading profile",
        trading=TradingConfig(symbols=["BTC-USD", "ETH-USD"], mode="normal"),
        risk=RiskConfig(
            max_position_size=Decimal("50000"),
            max_leverage=1,
            enable_shorts=False,
        ),
        execution=ExecutionConfig(
            dry_run=False,
            mock_broker=False,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
    ),
    Profile.CANARY: ProfileSchema(
        profile_name="canary",
        environment="production",
        description="Ultra-conservative production testing profile",
        trading=TradingConfig(symbols=["BTC-USD"], mode="reduce_only"),
        risk=RiskConfig(
            max_position_size=Decimal("500"),
            max_leverage=1,
            enable_shorts=True,  # But reduce_only mode
            daily_loss_limit_pct=0.01,
        ),
        execution=ExecutionConfig(
            time_in_force="IOC",
            dry_run=False,
            mock_broker=False,
            use_limit_orders=True,
            market_order_fallback=False,
        ),
        session=SessionConfig(
            start_time=time(14, 0),  # 2 PM UTC
            end_time=time(15, 0),  # 3 PM UTC
        ),
        monitoring=MonitoringConfig(log_level="DEBUG", update_interval=5),
    ),
    Profile.PROD: ProfileSchema(
        profile_name="prod",
        environment="production",
        description="Full production profile for perpetuals",
        trading=TradingConfig(symbols=["BTC-USD", "ETH-USD"], mode="normal"),
        risk=RiskConfig(
            max_position_size=Decimal("50000"),
            max_leverage=3,
            enable_shorts=True,
        ),
        execution=ExecutionConfig(
            dry_run=False,
            mock_broker=False,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
    ),
    Profile.PAPER: ProfileSchema(
        profile_name="paper",
        environment="paper",
        description="Paper trading profile with simulated execution",
        trading=TradingConfig(symbols=["BTC-USD", "ETH-USD"], mode="normal"),
        risk=RiskConfig(
            max_position_size=Decimal("10000"),
            max_leverage=3,
            enable_shorts=True,
            daily_loss_limit_pct=0.05,
        ),
        execution=ExecutionConfig(
            dry_run=True,
            mock_broker=True,
            mock_fills=True,
            use_limit_orders=False,
            market_order_fallback=True,
        ),
        monitoring=MonitoringConfig(log_level="INFO"),
    ),
}


class ProfileLoader:
    """Loads profile configuration from YAML files with hardcoded fallbacks.

    Usage:
        loader = ProfileLoader()
        schema = loader.load(Profile.DEV)
        bot_config = loader.to_bot_config(schema, Profile.DEV)
    """

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self.profiles_dir = profiles_dir or (path_registry.PROJECT_ROOT / "config" / "profiles")

    def load(self, profile: Profile) -> ProfileSchema:
        """Load profile schema from YAML file or fall back to defaults.

        Args:
            profile: The profile enum to load

        Returns:
            ProfileSchema with merged YAML + defaults
        """
        yaml_path = self.profiles_dir / f"{profile.value}.yaml"

        # Start with hardcoded defaults
        defaults = _PROFILE_DEFAULTS.get(profile)
        if defaults is None:
            defaults = ProfileSchema(profile_name=profile.value)

        # Try to load and merge YAML
        if yaml_path.exists():
            try:
                import yaml

                with yaml_path.open("r") as f:
                    yaml_data = yaml.safe_load(f) or {}

                schema = ProfileSchema.from_yaml(yaml_data, profile.value)

                logger.info(
                    "Loaded profile from YAML",
                    operation="profile_load",
                    profile=profile.value,
                    path=str(yaml_path),
                )

                return schema

            except Exception as e:
                logger.warning(
                    "Failed to load profile YAML, using defaults",
                    operation="profile_load",
                    profile=profile.value,
                    error=str(e),
                )

        return defaults

    def to_bot_config_kwargs(self, schema: ProfileSchema, profile: Profile) -> dict[str, Any]:
        """Convert ProfileSchema to BotConfig constructor kwargs.

        Args:
            schema: The loaded profile schema
            profile: The profile enum

        Returns:
            Dictionary of kwargs for BotConfig constructor
        """
        from gpt_trader.app.config.bot_config import BotRiskConfig
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            PerpsStrategyConfig,
        )

        # Build BotRiskConfig from schema.risk
        risk_config = BotRiskConfig(
            max_leverage=schema.risk.max_leverage,
            max_position_size=schema.risk.max_position_size,
            position_fraction=schema.risk.position_fraction,
            stop_loss_pct=schema.risk.stop_loss_pct,
            take_profit_pct=schema.risk.take_profit_pct,
            daily_loss_limit_pct=schema.risk.daily_loss_limit_pct,
        )

        # Build PerpsStrategyConfig from schema.strategy
        strategy_config = PerpsStrategyConfig(
            short_ma_period=schema.strategy.short_ma_period,
            long_ma_period=schema.strategy.long_ma_period,
            rsi_period=schema.strategy.rsi_period,
            rsi_overbought=schema.strategy.rsi_overbought,
            rsi_oversold=schema.strategy.rsi_oversold,
        )

        kwargs: dict[str, Any] = {
            "profile": profile,
            "symbols": schema.trading.symbols,
            "interval": schema.trading.interval,
            # Nested risk config (BotRiskConfig instance)
            "risk": risk_config,
            # Nested strategy config (PerpsStrategyConfig instance)
            "strategy": strategy_config,
            # Top-level config fields
            "enable_shorts": schema.risk.enable_shorts,
            # Execution settings
            "time_in_force": schema.execution.time_in_force,
            "dry_run": schema.execution.dry_run,
            "mock_broker": schema.execution.mock_broker,
            "use_limit_orders": schema.execution.use_limit_orders,
            "market_order_fallback": schema.execution.market_order_fallback,
            # Monitoring
            "log_level": schema.monitoring.log_level,
            "status_interval": schema.monitoring.update_interval,
            "status_enabled": schema.monitoring.status_enabled,
            # Strategy type
            "strategy_type": schema.strategy.type,
        }

        # Mode mapping
        if schema.trading.mode == "reduce_only":
            kwargs["reduce_only_mode"] = True

        # Note: Session time fields (trading_window_start, trading_window_end, trading_days)
        # are not currently in BotConfig. If needed, add them to BotConfig first.

        return kwargs


def get_profile_loader() -> ProfileLoader:
    """Get the ProfileLoader instance from the application container.

    Requires that an ApplicationContainer has been set via
    set_application_container(). This ensures proper dependency
    injection and avoids hidden global state.

    Returns:
        The ProfileLoader instance from the container.

    Raises:
        RuntimeError: If no application container has been set.
    """
    from gpt_trader.app.container import get_application_container

    container = get_application_container()
    if container is None:
        raise RuntimeError(
            "No application container set. Call set_application_container() "
            "before using get_profile_loader(). For tests, ensure the container "
            "is set up in a fixture."
        )
    return container.profile_loader


def load_profile(profile: Profile) -> ProfileSchema:
    """Convenience function to load a profile schema."""
    return get_profile_loader().load(profile)


ConfigFactory = Callable[..., "BotConfig"]


def build_profile_config(profile: Profile, create_config: ConfigFactory) -> BotConfig:
    """Construct a configuration tailored to the requested profile.

    Uses YAML-first loading: attempts to load from config/profiles/{profile}.yaml
    and falls back to hardcoded defaults if the file doesn't exist.

    Args:
        profile: The profile enum to load
        create_config: Factory function to create BotConfig

    Returns:
        BotConfig configured for the specified profile
    """
    # Load profile schema (YAML-first with hardcoded fallback)
    schema = load_profile(profile)

    # Convert schema to BotConfig
    return _schema_to_bot_config(schema, profile, create_config)


def _schema_to_bot_config(
    schema: ProfileSchema,
    profile: Profile,
    create_config: ConfigFactory,
) -> BotConfig:
    """Convert ProfileSchema to BotConfig using the factory.

    Args:
        schema: The loaded profile schema
        profile: The profile enum
        create_config: Factory function to create BotConfig

    Returns:
        BotConfig instance
    """
    from gpt_trader.app.config.bot_config import BotRiskConfig

    # Build risk config
    risk = BotRiskConfig(
        max_leverage=schema.risk.max_leverage,
        max_position_size=schema.risk.max_position_size,
        position_fraction=schema.risk.position_fraction,
        stop_loss_pct=schema.risk.stop_loss_pct,
        take_profit_pct=schema.risk.take_profit_pct,
    )

    # Base kwargs
    kwargs: dict = {
        "profile": profile,
        "symbols": schema.trading.symbols,
        "interval": schema.trading.interval,
        "risk": risk,
        "enable_shorts": schema.risk.enable_shorts,
        "time_in_force": schema.execution.time_in_force,
        "dry_run": schema.execution.dry_run,
        "mock_broker": schema.execution.mock_broker,
        "log_level": schema.monitoring.log_level,
        "status_interval": schema.monitoring.update_interval,
        "status_enabled": schema.monitoring.status_enabled,
        "strategy_type": schema.strategy.type,
    }

    # Mode mapping
    if schema.trading.mode == "reduce_only":
        kwargs["reduce_only_mode"] = True

    # Optional session settings
    if schema.session.start_time is not None:
        kwargs["trading_window_start"] = schema.session.start_time
    if schema.session.end_time is not None:
        kwargs["trading_window_end"] = schema.session.end_time
    if schema.session.trading_days:
        kwargs["trading_days"] = schema.session.trading_days

    return create_config(**kwargs)


__all__ = [
    "ExecutionConfig",
    "MonitoringConfig",
    "ProfileLoader",
    "ProfileSchema",
    "RiskConfig",
    "SessionConfig",
    "StrategyConfig",
    "TradingConfig",
    "build_profile_config",
    "get_profile_loader",
    "load_profile",
]
