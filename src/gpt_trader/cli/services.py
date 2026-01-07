"""Helper utilities for CLI command implementations."""

from argparse import Namespace
from dataclasses import fields as dataclass_fields
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    create_application_container,
    get_application_container,
    set_application_container,
)
from gpt_trader.orchestration.trading_bot.bot import TradingBot
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli_services")


def _filter_dataclass_fields(data: dict[str, Any], dataclass_type: type) -> dict[str, Any]:
    """Filter dict to only include keys that are valid dataclass fields."""
    valid_fields = {f.name for f in dataclass_fields(dataclass_type)}
    return {k: v for k, v in data.items() if k in valid_fields}


def _coerce_decimal_fields(data: dict[str, Any], dataclass_type: type) -> dict[str, Any]:
    """Convert numeric values to Decimal for fields that expect Decimal."""
    from dataclasses import fields as dc_fields

    result = dict(data)
    for f in dc_fields(dataclass_type):
        if f.name in result and f.type in (Decimal, "Decimal"):
            val = result[f.name]
            if val is not None and not isinstance(val, Decimal):
                result[f.name] = Decimal(str(val))
    return result


def load_config_from_yaml(path: str | Path) -> BotConfig:
    """Load BotConfig from a nested YAML file (e.g., optimize apply output).

    Supports structure:
        strategy:
            short_ma_period: 8
            long_ma_period: 35
            ...
        risk:
            target_leverage: 5
            position_fraction: 0.2
            ...
        symbols: [BTC-USD]
        interval: 60
        ...
    """
    from gpt_trader.features.live_trade.strategies.perps_baseline import (
        PerpsStrategyConfig,
    )

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Build strategy config from nested section
    strategy_data = data.get("strategy", {})
    strategy_kwargs = _filter_dataclass_fields(strategy_data, PerpsStrategyConfig)
    strategy = PerpsStrategyConfig(**strategy_kwargs)

    # Build risk config from nested section
    risk_data = data.get("risk", {})
    risk_kwargs = _filter_dataclass_fields(risk_data, BotRiskConfig)
    risk_kwargs = _coerce_decimal_fields(risk_kwargs, BotRiskConfig)
    risk = BotRiskConfig(**risk_kwargs)

    # Build BotConfig with nested configs + top-level fields
    return BotConfig(
        strategy=strategy,
        risk=risk,
        symbols=data.get("symbols", ["BTC-USD", "ETH-USD"]),
        interval=data.get("interval", 60),
        derivatives_enabled=data.get("derivatives_enabled", False),
        enable_shorts=data.get("enable_shorts", False),
        reduce_only_mode=data.get("reduce_only_mode", False),
        time_in_force=data.get("time_in_force", "GTC"),
        enable_order_preview=data.get("enable_order_preview", False),
        account_telemetry_interval=data.get("account_telemetry_interval"),
        log_level=data.get("log_level", "INFO"),
        dry_run=data.get("dry_run", False),
        mock_broker=data.get("mock_broker", False),
        profile=data.get("profile"),
        metadata=data.get("metadata", {}),
    )


def build_config_from_args(args: Namespace, **kwargs: Any) -> BotConfig:
    """
    Build configuration from environment, profile, config file, and CLI arguments.
    Precedence: CLI Args > Config File > Profile > Environment > Defaults

    Uses ProfileLoader to properly load all profile settings including:
    - trading: symbols, interval, mode
    - risk_management: leverage, position sizing, loss limits
    - strategy: type, MA periods
    - monitoring: log level, status settings
    - execution: dry_run, mock_broker, time_in_force
    """
    from gpt_trader.config.types import Profile
    from gpt_trader.orchestration.configuration.profile_loader import ProfileLoader

    # 1. Check for --config flag first (takes precedence over profile)
    config_path = getattr(args, "config", None)
    if config_path:
        config = load_config_from_yaml(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        # Start with Env/Defaults
        config = BotConfig.from_env()

        # Load Profile if specified using ProfileLoader
        profile_name = getattr(args, "profile", "dev")

        try:
            # Try to convert profile name to Profile enum
            profile_enum = Profile(profile_name)

            # Use ProfileLoader to load all profile settings
            loader = ProfileLoader()
            schema = loader.load(profile_enum)
            profile_kwargs = loader.to_bot_config_kwargs(schema, profile_enum)

            # Apply profile settings to config
            # Risk config (nested BotRiskConfig)
            if "risk" in profile_kwargs:
                config.risk = profile_kwargs["risk"]

            # Strategy config (nested PerpsStrategyConfig)
            if "strategy" in profile_kwargs:
                config.strategy = profile_kwargs["strategy"]

            # Trading settings
            if "symbols" in profile_kwargs:
                config.symbols = profile_kwargs["symbols"]
            if "interval" in profile_kwargs:
                config.interval = profile_kwargs["interval"]

            # Execution settings
            if "dry_run" in profile_kwargs:
                config.dry_run = profile_kwargs["dry_run"]
            if "mock_broker" in profile_kwargs:
                config.mock_broker = profile_kwargs["mock_broker"]
            if "time_in_force" in profile_kwargs:
                config.time_in_force = profile_kwargs["time_in_force"]

            # Other settings
            if "enable_shorts" in profile_kwargs:
                config.enable_shorts = profile_kwargs["enable_shorts"]
            if "reduce_only_mode" in profile_kwargs:
                config.reduce_only_mode = profile_kwargs["reduce_only_mode"]
            if "strategy_type" in profile_kwargs:
                config.strategy_type = profile_kwargs["strategy_type"]
            if "log_level" in profile_kwargs:
                config.log_level = profile_kwargs["log_level"]
            if "status_interval" in profile_kwargs:
                config.status_interval = profile_kwargs["status_interval"]
            if "status_enabled" in profile_kwargs:
                config.status_enabled = profile_kwargs["status_enabled"]
            if "profile" in profile_kwargs:
                config.profile = profile_kwargs["profile"]

            logger.info("Loaded profile %s with risk/strategy/monitoring settings", profile_name)

        except ValueError:
            # Invalid profile enum, try loading as raw YAML file
            profile_path = Path(f"config/profiles/{profile_name}.yaml")
            if profile_path.exists():
                try:
                    with open(profile_path) as f:
                        profile_data = yaml.safe_load(f)

                    # Map profile fields to BotConfig (legacy fallback)
                    trading = profile_data.get("trading", {})
                    if "symbols" in trading:
                        config.symbols = trading["symbols"]

                    execution = profile_data.get("execution", {})
                    if "mock_broker" in execution:
                        config.mock_broker = execution["mock_broker"]
                    if "dry_run" in execution:
                        config.dry_run = execution["dry_run"]

                    logger.warning(
                        "Profile %s loaded with legacy fallback (limited settings)", profile_name
                    )
                except Exception as e:
                    logger.warning("Failed to load profile %s: %s", profile_name, e)
        except Exception as e:
            logger.warning("Failed to load profile %s: %s", profile_name, e)

    # 2. Override with CLI Args (always takes highest precedence)
    if getattr(args, "dry_run", False):
        config.dry_run = True

    if getattr(args, "symbols", None):
        config.symbols = args.symbols

    if getattr(args, "interval", None):
        config.interval = args.interval

    if getattr(args, "target_leverage", None):
        # Update nested risk config
        config.risk.target_leverage = args.target_leverage

    if getattr(args, "reduce_only_mode", False):
        config.reduce_only_mode = True

    if getattr(args, "time_in_force", None):
        config.time_in_force = args.time_in_force

    if getattr(args, "enable_order_preview", False):
        config.enable_order_preview = True

    if getattr(args, "account_telemetry_interval", None):
        config.account_telemetry_interval = args.account_telemetry_interval

    return config


def instantiate_bot(config: BotConfig) -> TradingBot:
    """Instantiate a TradingBot using the ApplicationContainer.

    Registers the container globally so services can resolve dependencies
    via get_application_container(). Avoids overriding an existing container
    (e.g., one set by tests).
    """
    # Check if container already set (e.g., by tests)
    existing = get_application_container()
    if existing is not None:
        logger.debug("Using existing application container")
        return existing.create_bot()

    # Create and register new container
    container = create_application_container(config)
    set_application_container(container)
    logger.debug("Created and registered application container")
    return container.create_bot()
