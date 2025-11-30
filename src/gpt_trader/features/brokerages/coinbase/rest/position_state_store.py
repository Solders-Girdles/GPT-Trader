"""Centralized position state management for PnL tracking."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from gpt_trader.features.brokerages.coinbase.utilities import PositionState


class PositionStateStore:
    """Thread-safe storage for position state used in PnL calculations.

    This class provides centralized management of the _positions dict
    that was previously shared between CoinbaseRestServiceBase and PnLRestMixin.

    Thread Safety:
        All methods are protected by an RLock to ensure safe concurrent access
        from WebSocket threads and main trading threads.

    The `all()` method returns a defensive copy to prevent external
    modification of internal state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._positions: dict[str, PositionState] = {}

    def get(self, symbol: str) -> PositionState | None:
        """Get position state for a symbol."""
        with self._lock:
            return self._positions.get(symbol)

    def set(self, symbol: str, position: PositionState) -> None:
        """Set position state for a symbol."""
        with self._lock:
            self._positions[symbol] = position

    def contains(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        with self._lock:
            return symbol in self._positions

    def symbols(self) -> Iterator[str]:
        """Iterate over all symbols with positions.

        Returns a copy of the keys to allow safe iteration during modification.
        """
        with self._lock:
            return iter(list(self._positions.keys()))

    def all(self) -> dict[str, PositionState]:
        """Get all positions.

        Returns a defensive copy to prevent external modification.
        For backward compatibility with code that modifies the returned dict,
        use `set()` to persist changes back.
        """
        with self._lock:
            return dict(self._positions)

    def remove(self, symbol: str) -> None:
        """Remove position for a symbol."""
        with self._lock:
            self._positions.pop(symbol, None)

    def clear(self) -> None:
        """Clear all positions."""
        with self._lock:
            self._positions.clear()

    def __len__(self) -> int:
        """Return number of positions."""
        with self._lock:
            return len(self._positions)

    def __contains__(self, symbol: str) -> bool:
        """Support `in` operator."""
        return self.contains(symbol)
