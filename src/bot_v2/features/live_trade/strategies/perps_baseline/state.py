"""Lightweight state container for the baseline strategy."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from decimal import Decimal

from bot_v2.features.live_trade.strategies.shared import update_trailing_stop


@dataclass
class StrategyState:
    """Tracks per-symbol state such as trailing stops and position adds."""

    position_adds: MutableMapping[str, int] = field(default_factory=dict)
    trailing_stops: MutableMapping[str, tuple[Decimal, Decimal]] = field(default_factory=dict)

    def reset(self, symbol: str | None = None) -> None:
        """Reset state for a single symbol or the entire strategy."""
        if symbol is not None:
            self.position_adds.pop(symbol, None)
            self.trailing_stops.pop(symbol, None)
            return
        self.position_adds.clear()
        self.trailing_stops.clear()

    def update_trailing_stop(
        self,
        *,
        symbol: str,
        side: str,
        current_price: Decimal,
        trailing_pct: Decimal,
    ) -> bool:
        """Update trailing stop tracking and return True if it fired."""
        return update_trailing_stop(
            self.trailing_stops,
            symbol=symbol,
            side=side,
            current_price=current_price,
            trailing_pct=trailing_pct,
        )


__all__ = ["StrategyState"]
