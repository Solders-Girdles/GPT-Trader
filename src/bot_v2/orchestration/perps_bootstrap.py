"""Bootstrap helpers for preparing PerpsBot dependencies."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore

from .configuration import TOP_VOLUME_BASES, BotConfig, Profile
from .service_registry import ServiceRegistry, empty_registry

_DEFAULT_ALLOWED_PERPS: frozenset[str] = frozenset(
    {
        "BTC-PERP",
        "ETH-PERP",
        "SOL-PERP",
        "XRP-PERP",
    }
)


@dataclass(frozen=True)
class BootstrapLogRecord:
    """Captured log entry emitted during bootstrap."""

    level: int
    message: str
    args: tuple[object, ...] = ()


@dataclass(frozen=True)
class RuntimePaths:
    """Materialized runtime directories for the bot."""

    storage_dir: Path
    event_store_root: Path


@dataclass(frozen=True)
class PerpsBootstrapResult:
    """Outcome of preparing dependencies for :class:`PerpsBot`."""

    config: BotConfig
    registry: ServiceRegistry
    runtime_paths: RuntimePaths
    event_store: EventStore
    orders_store: OrdersStore
    logs: list[BootstrapLogRecord]


def normalise_symbols(
    requested: Sequence[str] | None,
    *,
    derivatives_enabled: bool,
    default_quote: str,
    allowed_perps: Iterable[str] = _DEFAULT_ALLOWED_PERPS,
    fallback_bases: Sequence[str] = TOP_VOLUME_BASES,
) -> tuple[list[str], list[BootstrapLogRecord]]:
    """Return canonical symbol list for the configured runtime."""

    logs: list[BootstrapLogRecord] = []
    allowed_set = set(allowed_perps)
    normalised: list[str] = []

    for raw in requested or []:
        symbol = (raw or "").strip().upper()
        if not symbol:
            continue

        if derivatives_enabled:
            if symbol not in allowed_set:
                logs.append(
                    BootstrapLogRecord(
                        logging.WARNING,
                        "Filtering unsupported perpetual symbol %s. Allowed perps: %s",
                        (symbol, sorted(allowed_set)),
                    )
                )
                continue
            normalised.append(symbol)
            continue

        if symbol.endswith("-PERP"):
            base = symbol.split("-", 1)[0]
            replacement = f"{base}-{default_quote.upper()}"
            logs.append(
                BootstrapLogRecord(
                    logging.WARNING,
                    "Derivatives disabled. Replacing %s with spot symbol %s",
                    (symbol, replacement),
                )
            )
            symbol = replacement

        normalised.append(symbol)

    # Deduplicate while preserving the first occurrence order
    normalised = list(dict.fromkeys(normalised))

    if normalised:
        return normalised, logs

    if derivatives_enabled:
        fallback = ["BTC-PERP", "ETH-PERP"]
    else:
        quote = default_quote.upper()
        fallback = [f"{base}-{quote}" for base in fallback_bases]

    logs.append(
        BootstrapLogRecord(
            logging.INFO,
            "No valid symbols provided. Falling back to %s",
            (fallback,),
        )
    )
    return fallback, logs


def resolve_runtime_paths(
    profile: Profile,
    env: Mapping[str, str] | None = None,
) -> RuntimePaths:
    """Determine and materialise storage directories for the bot."""

    env_map: Mapping[str, str] = env or os.environ

    runtime_root = Path(env_map.get("GPT_TRADER_RUNTIME_ROOT", str(RUNTIME_DATA_DIR)))
    storage_dir = runtime_root / f"perps_bot/{profile.value}"
    storage_dir.mkdir(parents=True, exist_ok=True)

    event_root = env_map.get("EVENT_STORE_ROOT")
    if event_root:
        event_store_root = Path(event_root)
        if "perps_bot" not in set(event_store_root.parts):
            event_store_root = event_store_root / "perps_bot" / profile.value
    else:
        event_store_root = storage_dir

    event_store_root.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(storage_dir=storage_dir, event_store_root=event_store_root)


def prepare_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> PerpsBootstrapResult:
    """Prepare directories and registry dependencies for :class:`PerpsBot`."""

    env_map: Mapping[str, str]
    if env is None:
        env_map = os.environ
    else:
        env_map = env

    derivatives_enabled = (
        config.profile != Profile.SPOT and env_map.get("COINBASE_ENABLE_DERIVATIVES", "0") == "1"
    )
    default_quote = env_map.get("COINBASE_DEFAULT_QUOTE", "USD").upper()

    symbols, logs = normalise_symbols(
        config.symbols,
        derivatives_enabled=derivatives_enabled,
        default_quote=default_quote,
    )
    config.symbols = list(symbols)

    runtime_paths = resolve_runtime_paths(config.profile, env_map)

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

    return PerpsBootstrapResult(
        config=config,
        registry=prepared_registry,
        runtime_paths=runtime_paths,
        event_store=event_store,
        orders_store=orders_store,
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
