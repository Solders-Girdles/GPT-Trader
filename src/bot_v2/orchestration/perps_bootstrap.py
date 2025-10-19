"""Bootstrap helpers for preparing PerpsBot dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore

from .configuration import TOP_VOLUME_BASES, BotConfig, Profile
from .runtime_paths import RuntimePaths
from .runtime_paths import resolve_runtime_paths as compute_runtime_paths
from .runtime_settings import RuntimeSettings, load_runtime_settings
from .service_registry import ServiceRegistry, empty_registry
from .symbols import PERPS_ALLOWLIST, normalize_symbol_list


@dataclass(frozen=True)
class BootstrapLogRecord:
    """Captured log entry emitted during bootstrap."""

    level: int
    message: str
    args: tuple[object, ...] = ()


@dataclass(frozen=True)
class PerpsBootstrapResult:
    """Outcome of preparing dependencies for :class:`PerpsBot`."""

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


def prepare_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    env: Mapping[str, str] | None = None,
    settings: RuntimeSettings | None = None,
) -> PerpsBootstrapResult:
    """Prepare directories and registry dependencies for :class:`PerpsBot`."""

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

    runtime_paths = resolve_runtime_paths(config.profile, settings)

    prepared_registry = registry or empty_registry(config)
    if prepared_registry.config is not config:
        prepared_registry = prepared_registry.with_updates(config=config)

    event_store = prepared_registry.event_store
    if event_store is None:
        event_store = EventStore(root=runtime_paths.event_store_root)
        prepared_registry = prepared_registry.with_updates(event_store=event_store)

    orders_store = prepared_registry.orders_store
    if orders_store is None:
        orders_store = OrdersStore(storage_path=runtime_paths.storage_dir)
        prepared_registry = prepared_registry.with_updates(orders_store=orders_store)

    prepared_registry = prepared_registry.with_updates(runtime_settings=settings)

    return PerpsBootstrapResult(
        config=config,
        registry=prepared_registry,
        runtime_paths=runtime_paths,
        event_store=event_store,
        orders_store=orders_store,
        settings=settings,
        logs=logs,
    )


__all__ = [
    "BootstrapLogRecord",
    "PerpsBootstrapResult",
    "prepare_perps_bot",
    "normalise_symbols",
    "resolve_runtime_paths",
    "RuntimePaths",
]
