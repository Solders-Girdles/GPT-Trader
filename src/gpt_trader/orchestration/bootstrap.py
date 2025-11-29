"""Bootstrap helpers for preparing TradingBot dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore
from gpt_trader.utilities.logging_patterns import get_logger

from .configuration import TOP_VOLUME_BASES, BotConfig, Profile
from .runtime_paths import RuntimePaths
from .runtime_paths import resolve_runtime_paths as compute_runtime_paths
from .service_registry import ServiceRegistry, empty_registry
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
    settings: RuntimeSettings
    logs: list[BootstrapLogRecord]


def normalise_symbols(
    requested: Sequence[str] | None,
    *,
    settings: RuntimeSettings,
    allowed_perps: Iterable[str] = PERPS_ALLOWLIST,
    fallback_bases: Sequence[str] = TOP_VOLUME_BASES,
) -> tuple[list[str], list[BootstrapLogRecord]]:
    """Return canonical symbol list for the configured runtime."""

    symbol_quote = settings.coinbase_default_quote
    normalised, records = normalize_symbol_list(
        requested,
        allow_derivatives=settings.coinbase_enable_derivatives,
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
    settings: RuntimeSettings,
) -> RuntimePaths:
    """Determine and materialise storage directories for the bot."""

    return compute_runtime_paths(settings=settings, profile=profile)


def prepare_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    env: Mapping[str, str] | None = None,
    settings: RuntimeSettings | None = None,
) -> BootstrapResult:
    """Prepare directories and registry dependencies for :class:`TradingBot`."""

    if settings is None:
        if registry is not None and registry.runtime_settings is not None:
            settings = registry.runtime_settings
        else:
            settings = load_runtime_settings(env)

    metadata = dict(config.metadata)
    existing_overrides_raw = metadata.get("symbol_normalization_overrides")
    overrides = dict(existing_overrides_raw) if isinstance(existing_overrides_raw, dict) else {}
    metadata_changed = False

    if settings.coinbase_default_quote_overridden:
        if overrides.get("quote") != settings.coinbase_default_quote:
            overrides["quote"] = settings.coinbase_default_quote
            metadata_changed = True

    if settings.coinbase_enable_derivatives_overridden:
        allow_override = settings.coinbase_enable_derivatives
        if overrides.get("allow_derivatives") != allow_override:
            overrides["allow_derivatives"] = allow_override
            metadata_changed = True

    cleaned_overrides = {key: value for key, value in overrides.items() if value is not None}

    current_overrides = (
        dict(existing_overrides_raw) if isinstance(existing_overrides_raw, dict) else {}
    )

    if cleaned_overrides:
        if current_overrides != cleaned_overrides:
            metadata["symbol_normalization_overrides"] = cleaned_overrides
            metadata_changed = True
    else:
        if existing_overrides_raw is not None:
            metadata.pop("symbol_normalization_overrides", None)
            metadata_changed = True

    if metadata_changed:
        rebuilt = config.model_copy(update={"metadata": metadata})
        for field_name in config.model_fields:
            setattr(config, field_name, getattr(rebuilt, field_name))
        metadata = dict(config.metadata)

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
        _, fallback_logs = normalise_symbols(config.symbols, settings=settings)
        logs = [
            BootstrapLogRecord(level=record.level, message=record.message, args=record.args)
            for record in fallback_logs
        ]

    runtime_paths = resolve_runtime_paths(cast(Profile, config.profile), settings)

    prepared_registry = registry or empty_registry(config)
    if prepared_registry.config is not config:
        prepared_registry = prepared_registry.with_updates(config=config)

    # Get or create event store (using concrete type for BootstrapResult)
    registry_event_store = prepared_registry.event_store
    if registry_event_store is None:
        final_event_store: EventStore = EventStore(root=runtime_paths.event_store_root)
        prepared_registry = prepared_registry.with_updates(event_store=final_event_store)
    else:
        # Registry may hold a protocol; cast to concrete type
        final_event_store = cast(EventStore, registry_event_store)

    # Get or create orders store (using concrete type for BootstrapResult)
    registry_orders_store = prepared_registry.orders_store
    if registry_orders_store is None:
        final_orders_store: OrdersStore = OrdersStore(storage_path=runtime_paths.storage_dir)
        prepared_registry = prepared_registry.with_updates(orders_store=final_orders_store)
    else:
        final_orders_store = cast(OrdersStore, registry_orders_store)

    prepared_registry = prepared_registry.with_updates(runtime_settings=settings)

    result = BootstrapResult(
        config=config,
        registry=prepared_registry,
        runtime_paths=runtime_paths,
        event_store=final_event_store,
        orders_store=final_orders_store,
        settings=settings,
        logs=logs,
    )

    from gpt_trader.app.container import create_application_container

    container = create_application_container(config, settings)
    # Add container to result extras
    result.registry.extras["container"] = container
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
    settings: RuntimeSettings | None = None,
) -> BootstrapResult:
    """Prepare directories and registry dependencies for :class:`TradingBot` using container."""
    return prepare_bot(config, registry=None, env=env, settings=settings)


def build_bot(
    config: BotConfig,
    settings: RuntimeSettings | None = None,
) -> tuple[TradingBot, ServiceRegistry]:
    """
    Build a TradingBot from a BotConfig.

    Returns:
        Tuple of (TradingBot, ServiceRegistry) for access to all components.
    """
    from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
    from gpt_trader.monitoring.notifications import create_notification_service
    from gpt_trader.orchestration.trading_bot.bot import TradingBot

    result = prepare_bot(config, registry=None, settings=settings)

    # Get the container from extras (created by prepare_bot)
    container = result.registry.extras.get("container")

    # Get broker from container
    broker = container.broker if container else None

    # Create risk manager with config and event store
    risk_manager = LiveRiskManager(config=result.config, event_store=result.event_store)

    # Create notification service with webhook if configured
    notification_service = create_notification_service(
        webhook_url=config.webhook_url,
        console_enabled=True,
    )
    if config.webhook_url:
        logger.info(
            "Webhook notifications enabled",
            operation="bot_bootstrap",
            stage="notifications_enabled",
        )

    # Update registry with broker, risk manager, and notification service
    registry = result.registry.with_updates(
        broker=broker,
        risk_manager=risk_manager,
        notification_service=notification_service,
    )

    # Create the bot with full registry
    bot = TradingBot(
        config=result.config,
        container=container,
        registry=registry,
        event_store=result.event_store,
        orders_store=result.orders_store,
        notification_service=notification_service,
    )

    return bot, registry


def bot_from_profile(profile: str) -> tuple[TradingBot, ServiceRegistry]:
    """
    Create a TradingBot from a profile name.

    Args:
        profile: One of 'dev', 'demo', 'prod', 'test', 'spot', 'canary'

    Returns:
        Tuple of (TradingBot, ServiceRegistry)
    """
    # Convert string to Profile enum
    profile_enum = Profile(profile.lower())

    # Create config with mock broker for dev/test profiles
    mock_broker = profile_enum in (Profile.DEV, Profile.TEST)
    config = BotConfig.from_profile(
        profile=profile_enum,
        mock_broker=mock_broker,
    )

    # Load settings with mock mode for dev/test
    env_overrides: dict[str, str] = {}
    if mock_broker:
        env_overrides["PERPS_FORCE_MOCK"] = "1"

    settings = load_runtime_settings(env_overrides)

    return build_bot(config, settings=settings)


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
