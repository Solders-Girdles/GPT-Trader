"""Bootstrap helpers for preparing TradingBot dependencies.

This module provides convenience functions for creating TradingBot instances.
All functions use ``ApplicationContainer`` internally as the canonical
composition root.

Usage::

    from gpt_trader.app.container import ApplicationContainer
    container = ApplicationContainer(config)
    bot = container.create_bot()

    # Or use convenience functions:
    from gpt_trader.orchestration.bootstrap import build_bot, bot_from_profile
    bot = build_bot(config)
    bot = bot_from_profile("demo")
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

from .configuration import TOP_VOLUME_BASES, BotConfig, Profile
from .runtime_paths import RuntimePaths
from .runtime_paths import resolve_runtime_paths as compute_runtime_paths
from .symbols import PERPS_ALLOWLIST, normalize_symbol_list

if TYPE_CHECKING:
    from .trading_bot import TradingBot

logger = get_logger(__name__, component="bot_bootstrap")


@dataclass(frozen=True)
class BootstrapLogRecord:
    """Captured log entry emitted during bootstrap."""

    level: int
    message: str
    args: tuple[object, ...] = ()


def normalise_symbols(
    requested: Sequence[str] | None,
    *,
    config: BotConfig,
    allowed_perps: Iterable[str] = PERPS_ALLOWLIST,
    fallback_bases: Sequence[str] = TOP_VOLUME_BASES,
) -> tuple[list[str], list[BootstrapLogRecord]]:
    """Return canonical symbol list for the configured runtime."""

    symbol_quote = config.coinbase_default_quote
    normalised, records = normalize_symbol_list(
        requested,
        allow_derivatives=config.derivatives_enabled,
        quote=symbol_quote,
        allowed_perps=allowed_perps,
        fallback_bases=fallback_bases,
    )
    logs = [
        BootstrapLogRecord(level=record.level, message=record.message, args=record.args)
        for record in records
    ]
    return normalised, logs


def resolve_runtime_paths(
    profile: Profile,
    config: BotConfig,
) -> RuntimePaths:
    """Determine and materialise storage directories for the bot."""

    return compute_runtime_paths(config=config, profile=profile)


def build_bot(config: BotConfig) -> TradingBot:
    """Build a TradingBot from a BotConfig using ApplicationContainer.

    This is the canonical way to create a TradingBot. The container
    handles all dependency wiring and is registered globally for
    service resolution.

    Args:
        config: The bot configuration.

    Returns:
        A fully configured TradingBot ready to run.
    """
    from gpt_trader.app.container import (
        ApplicationContainer,
        set_application_container,
    )

    container = ApplicationContainer(config)
    set_application_container(container)

    if config.webhook_url:
        logger.info(
            "Webhook notifications enabled",
            operation="bot_bootstrap",
            stage="notifications_enabled",
        )

    return container.create_bot()


def bot_from_profile(profile: str) -> TradingBot:
    """Create a TradingBot from a profile name.

    Args:
        profile: One of 'dev', 'demo', 'prod', 'test', 'spot', 'canary'

    Returns:
        A fully configured TradingBot ready to run.
    """
    # Convert string to Profile enum
    profile_enum = Profile(profile.lower())

    # Create config with mock broker for dev/test profiles
    mock_broker = profile_enum in (Profile.DEV, Profile.TEST)
    config = BotConfig.from_profile(
        profile=profile_enum,
        mock_broker=mock_broker,
    )

    return build_bot(config)


__all__ = [
    "BootstrapLogRecord",
    "normalise_symbols",
    "resolve_runtime_paths",
    "RuntimePaths",
    "build_bot",
    "bot_from_profile",
]
