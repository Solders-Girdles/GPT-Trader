"""Profile-specific configuration builders.

Provides YAML-first profile loading with hardcoded fallbacks.
Profile YAML files are located in config/profiles/{profile_name}.yaml.

The loading priority is:
1. YAML file (if exists)
2. Hardcoded defaults (fallback)
3. CLI/environment overrides (applied after)
"""

from __future__ import annotations

from collections.abc import Callable

from gpt_trader.config.types import Profile
from gpt_trader.utilities.logging_patterns import get_logger

from .bot_config import BotConfig, BotRiskConfig
from .profile_loader import ProfileLoader, ProfileSchema, load_profile

ConfigFactory = Callable[..., BotConfig]

logger = get_logger(__name__, component="config_profiles")


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

    # Optional daily loss limit
    if schema.risk.daily_loss_limit is not None:
        kwargs["daily_loss_limit"] = schema.risk.daily_loss_limit

    return create_config(**kwargs)


__all__ = [
    "ProfileLoader",
    "ProfileSchema",
    "build_profile_config",
    "load_profile",
]
