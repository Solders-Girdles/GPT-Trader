"""
Protocol definitions for orchestration abstractions.

These protocols define the expected interfaces for service registry
and runtime state, enabling structural typing and better testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.orchestration.configuration import BotConfig


@runtime_checkable
class EventStoreProtocol(Protocol):
    """Protocol for event storage implementations."""

    def store(self, event: Any) -> None:
        """Store an event."""
        ...

    def get_recent(self, count: int = 100) -> list[Any]:
        """Get recent events."""
        ...


@runtime_checkable
class RuntimeStateProtocol(Protocol):
    """
    Protocol for runtime state management.

    Tracks active positions, equity, and trading state.
    """

    equity: Any
    positions: dict[str, Any]
    positions_pnl: dict[str, dict[str, Any]]
    positions_dict: dict[str, dict[str, Any]]

    def update_equity(self, value: Any) -> None:
        """Update current equity value."""
        ...


@runtime_checkable
class ServiceRegistryProtocol(Protocol):
    """
    Protocol for service registry implementations.

    Defines the interface for accessing shared services used by
    trading components. Implementations can be frozen dataclasses
    or dynamic container objects.
    """

    config: BotConfig
    broker: BrokerProtocol | None
    event_store: EventStoreProtocol | None
    orders_store: Any

    def with_updates(self, **kwargs: Any) -> ServiceRegistryProtocol:
        """Return a new registry with updated values."""
        ...


__all__ = [
    "EventStoreProtocol",
    "RuntimeStateProtocol",
    "ServiceRegistryProtocol",
]
