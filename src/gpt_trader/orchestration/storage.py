"""Helpers for preparing runtime storage used by live bots.

.. deprecated::
    This module is deprecated. Use `ApplicationContainer` from
    `gpt_trader.app.container` instead, which handles storage creation
    automatically through its `event_store` and `orders_store` properties.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore

if TYPE_CHECKING:
    from gpt_trader.app.container import ApplicationContainer


@dataclass(frozen=True)
class StorageContext:
    """Storage context containing persistence stores and paths."""

    event_store: EventStore
    orders_store: OrdersStore
    storage_dir: Path
    event_store_root: Path
    registry: ServiceRegistry


def create_storage_from_container(container: ApplicationContainer) -> StorageContext:
    """
    Create StorageContext from an ApplicationContainer.

    This is the preferred way to create storage. The container handles
    all dependency initialization correctly.

    Args:
        container: The application container.

    Returns:
        StorageContext with all storage components initialized.
    """
    return StorageContext(
        event_store=container.event_store,
        orders_store=container.orders_store,
        storage_dir=container.runtime_paths.storage_dir,
        event_store_root=container.runtime_paths.event_store_root,
        registry=container.create_service_registry(),
    )


class StorageBootstrapper:
    """
    Builds persistence collaborators for a bot instance.

    .. deprecated::
        Use `ApplicationContainer` instead. The container handles storage
        creation through its `event_store` and `orders_store` properties.
        For backward compatibility, use `create_storage_from_container()`.
    """

    def __init__(self, container: ApplicationContainer) -> None:
        warnings.warn(
            "StorageBootstrapper is deprecated. Use ApplicationContainer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._container = container

    def bootstrap(self) -> StorageContext:
        return create_storage_from_container(self._container)
