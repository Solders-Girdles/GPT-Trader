"""Helpers for preparing runtime storage used by live bots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore


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
        profile = self._config.profile.value
        runtime_root = Path(os.environ.get("GPT_TRADER_RUNTIME_ROOT", str(RUNTIME_DATA_DIR)))
        storage_dir = runtime_root / f"perps_bot/{profile}"
        storage_dir.mkdir(parents=True, exist_ok=True)

        event_root_env = os.environ.get("EVENT_STORE_ROOT")
        if event_root_env:
            event_store_root = Path(event_root_env)
            if "perps_bot" not in set(event_store_root.parts):
                event_store_root = event_store_root / "perps_bot" / profile
        else:
            event_store_root = storage_dir
        event_store_root.mkdir(parents=True, exist_ok=True)

        registry = self._registry

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
