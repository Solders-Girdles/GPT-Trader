"""Bootstrap helpers for preparing TradingBot dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore
from gpt_trader.utilities.logging_patterns import get_logger

from .configuration import TOP_VOLUME_BASES, BotConfig, Profile
from .runtime_paths import RuntimePaths
from .runtime_paths import resolve_runtime_paths as compute_runtime_paths
from .service_registry import ServiceRegistry
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


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of preparing dependencies for :class:`TradingBot`."""

    config: BotConfig
    registry: ServiceRegistry
    runtime_paths: RuntimePaths
    event_store: EventStore
    orders_store: OrdersStore
    logs: list[BootstrapLogRecord]


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


def prepare_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> BootstrapResult:
    """Prepare directories and registry dependencies for :class:`TradingBot`."""

    normalization_log_payload = config.metadata.get("symbol_normalization_logs", [])
    if normalization_log_payload:
        logs = [
            BootstrapLogRecord(
                level=int(entry.get("level", 20)),
                message=str(entry.get("message", "")),
                args=tuple(entry.get("args", ())),
            )
            for entry in normalization_log_payload
            if isinstance(entry, dict)
        ]
    else:
        _, fallback_logs = normalise_symbols(config.symbols, config=config)
        logs = [
            BootstrapLogRecord(level=record.level, message=record.message, args=record.args)
            for record in fallback_logs
        ]

    from gpt_trader.app.container import ApplicationContainer

    # Create container - this is now the source of truth
    container = ApplicationContainer(config)

    # Get registry from container (legacy support)
    # If a registry was passed in, we ignore it in favor of the container's registry
    # to ensure consistency, or we could try to merge, but container should be authoritative.
    # For now, we'll use the container's registry.
    prepared_registry = container.create_service_registry()

    # Add container to result extras for backward compatibility
    prepared_registry.extras["container"] = container

    # Get stores from container
    final_event_store = container.event_store
    final_orders_store = container.orders_store

    result = BootstrapResult(
        config=config,
        registry=prepared_registry,
        runtime_paths=container.runtime_paths,
        event_store=final_event_store,
        orders_store=final_orders_store,
        logs=logs,
    )

    logger.debug(
        "Prepared TradingBot with container",
        operation="bot_bootstrap",
        stage="container_enabled",
    )

    return result


def prepare_bot_with_container(
    config: BotConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> BootstrapResult:
    """Prepare directories and registry dependencies for :class:`TradingBot` using container."""
    return prepare_bot(config, registry=None, env=env)


def build_bot(config: BotConfig) -> TradingBot:
    """
    Build a TradingBot from a BotConfig using ApplicationContainer.

    This is the canonical way to create a TradingBot. The container
    handles all dependency wiring.

    Args:
        config: The bot configuration.

    Returns:
        A fully configured TradingBot ready to run.
    """
    from gpt_trader.app.container import ApplicationContainer

    container = ApplicationContainer(config)

    if config.webhook_url:
        logger.info(
            "Webhook notifications enabled",
            operation="bot_bootstrap",
            stage="notifications_enabled",
        )

    return container.create_bot()


def bot_from_profile(profile: str) -> TradingBot:
    """
    Create a TradingBot from a profile name.

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
    "BootstrapResult",
    "prepare_bot",
    "prepare_bot_with_container",
    "normalise_symbols",
    "resolve_runtime_paths",
    "RuntimePaths",
    "build_bot",
    "bot_from_profile",
]
