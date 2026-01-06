"""
Protocol definitions for orchestration abstractions.

These protocols define the expected interfaces for runtime state
and core services, enabling structural typing and better testability.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
class AccountManagerProtocol(Protocol):
    """Protocol for account manager implementations.

    Provides account snapshot and treasury operations.
    """

    def snapshot(self, emit_metric: bool = True) -> dict[str, Any]:
        """Collect account state snapshot."""
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

    # Strategy state
    strategy: Any  # BaselinePerpsStrategy for perps profile
    symbol_strategies: dict[str, Any]  # Per-symbol strategies for spot profile

    # Mark data state (for telemetry)
    mark_lock: Any  # threading.Lock
    mark_windows: dict[str, Any]  # Per-symbol mark price windows

    # Order book data state (for advanced strategies)
    orderbook_lock: Any  # threading.Lock for orderbook access
    orderbook_snapshots: dict[str, Any]  # Per-symbol DepthSnapshot

    # Trade flow data state (for volume analysis)
    trade_lock: Any  # threading.Lock for trade data access
    trade_aggregators: dict[str, Any]  # Per-symbol TradeTapeAgg

    def update_equity(self, value: Any) -> None:
        """Update current equity value."""
        ...


__all__ = [
    "AccountManagerProtocol",
    "EventStoreProtocol",
    "RuntimeStateProtocol",
]
