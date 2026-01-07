"""Persistence sub-container for ApplicationContainer.

This container manages persistence-related dependencies:
- Runtime paths (storage directories)
- Event store (order events, trade history)
- Orders store (order state persistence)
"""

from __future__ import annotations

from collections.abc import Callable

from gpt_trader.app.config import BotConfig
from gpt_trader.config.types import Profile
from gpt_trader.orchestration.runtime_paths import RuntimePaths, resolve_runtime_paths
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


class PersistenceContainer:
    """Container for persistence-related dependencies.

    This container lazily initializes runtime paths, event store, and orders
    store. It accepts a profile provider to support lazy resolution and avoid
    initialization order issues.

    Args:
        config: Bot configuration.
        profile_provider: Callable that returns the Profile instance.
            This is a callable (not the instance) to support lazy resolution.
    """

    def __init__(
        self,
        config: BotConfig,
        profile_provider: Callable[[], Profile],
    ):
        self._config = config
        self._profile_provider = profile_provider

        self._runtime_paths: RuntimePaths | None = None
        self._event_store: EventStore | None = None
        self._orders_store: OrdersStore | None = None

    @property
    def runtime_paths(self) -> RuntimePaths:
        """Resolve storage directories for the configured profile."""
        if self._runtime_paths is None:
            profile = self._profile_provider()
            self._runtime_paths = resolve_runtime_paths(
                config=self._config,
                profile=profile,
            )
        return self._runtime_paths

    @property
    def event_store(self) -> EventStore:
        """Get or create the event store."""
        if self._event_store is None:
            self._event_store = EventStore(root=self.runtime_paths.event_store_root)
        return self._event_store

    @property
    def orders_store(self) -> OrdersStore:
        """Get or create the orders store."""
        if self._orders_store is None:
            self._orders_store = OrdersStore(storage_path=self.runtime_paths.storage_dir)
        return self._orders_store
