"""Centralized position state management for PnL tracking."""

from __future__ import annotations

from collections.abc import Iterator

from gpt_trader.features.brokerages.coinbase.utilities import PositionState


class PositionStateStore:
    """Thread-safe storage for position state used in PnL calculations.

    This class provides centralized management of the _positions dict
    that was previously shared between CoinbaseRestServiceBase and PnLRestMixin.

    The `all()` method returns the internal dict reference for backward
    compatibility with code that accesses `service.positions` directly.
    """

    def __init__(self) -> None:
        self._positions: dict[str, PositionState] = {}

    def get(self, symbol: str) -> PositionState | None:
        """Get position state for a symbol."""
        return self._positions.get(symbol)

    def set(self, symbol: str, position: PositionState) -> None:
        """Set position state for a symbol."""
        self._positions[symbol] = position

    def contains(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self._positions

    def symbols(self) -> Iterator[str]:
        """Iterate over all symbols with positions.

        Returns a copy of the keys to allow safe iteration during modification.
        """
        return iter(list(self._positions.keys()))

    def all(self) -> dict[str, PositionState]:
        """Get all positions.

        Returns the internal dict reference for backward compatibility
        with code that accesses `service.positions` directly.
        """
        return self._positions

    def remove(self, symbol: str) -> None:
        """Remove position for a symbol."""
        self._positions.pop(symbol, None)

    def clear(self) -> None:
        """Clear all positions."""
        self._positions.clear()

    def __len__(self) -> int:
        """Return number of positions."""
        return len(self._positions)

    def __contains__(self, symbol: str) -> bool:
        """Support `in` operator."""
        return self.contains(symbol)
