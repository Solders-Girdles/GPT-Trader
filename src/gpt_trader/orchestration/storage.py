"""Helpers for preparing runtime storage used by live bots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.runtime_paths import resolve_runtime_paths
from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


@dataclass(frozen=True)
class StorageContext:
    event_store: EventStore
    orders_store: OrdersStore
    storage_dir: Path
    event_store_root: Path
    registry: ServiceRegistry


class StorageBootstrapper:
    """Builds persistence collaborators for a bot instance."""

    def __init__(self, config: BotConfig, registry: ServiceRegistry) -> None:
        self._config = config
        self._registry = registry

    def bootstrap(self) -> StorageContext:
        profile = self._config.profile.value  # type: ignore[attr-defined]
        settings = self._resolve_settings()
        runtime_paths = resolve_runtime_paths(settings=settings, profile=profile)
        storage_dir = runtime_paths.storage_dir
        event_store_root = runtime_paths.event_store_root

        registry = self._registry
        if registry.runtime_settings is None:
            registry = registry.with_updates(runtime_settings=settings)

        if registry.event_store is not None:
            event_store = registry.event_store
        else:
            event_store = EventStore(root=event_store_root)
            registry = registry.with_updates(event_store=event_store)

        if registry.orders_store is not None:
            orders_store = registry.orders_store
        else:
            orders_store = OrdersStore(storage_path=storage_dir)
            registry = registry.with_updates(orders_store=orders_store)

        return StorageContext(
            event_store=event_store,
            orders_store=orders_store,
            storage_dir=storage_dir,
            event_store_root=event_store_root,
            registry=registry,
        )

    def _resolve_settings(self) -> RuntimeSettings:
        registry_settings = self._registry.runtime_settings
        if registry_settings is not None:
            return registry_settings
        return load_runtime_settings()
