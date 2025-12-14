"""Trade matcher state management.

This module provides the state dataclass for trade matching operations,
separating state from the matching logic for better state management
and testability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.tui.types import Trade


@dataclass
class TradeMatcherState:
    """State for the trade matching algorithm.

    This dataclass holds all mutable state used by the trade matcher,
    enabling centralized state management and proper reset handling
    when mode changes occur.

    Attributes:
        unmatched_buys: Dict of symbol -> list of (Trade, remaining_quantity) pairs
            for buy trades that haven't been matched to a sell.
        unmatched_sells: Dict of symbol -> list of (Trade, remaining_quantity) pairs
            for sell trades that haven't been matched to a buy.
        pnl_accumulator: Dict of trade_id -> accumulated P&L value.
            Tracks realized P&L for partially and fully matched trades.
        processed_trade_ids: Set of trade IDs that have been processed.
            Used for incremental updates to avoid reprocessing.
        pnl_display_cache: Dict of trade_id -> formatted P&L display string.
            Caches the formatted display values for efficiency.
    """

    unmatched_buys: dict[str, list[tuple[Trade, Decimal]]] = field(default_factory=dict)
    unmatched_sells: dict[str, list[tuple[Trade, Decimal]]] = field(default_factory=dict)
    pnl_accumulator: dict[str, Decimal] = field(default_factory=dict)
    processed_trade_ids: set[str] = field(default_factory=set)
    pnl_display_cache: dict[str, str | None] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all state to initial values.

        Call this when the bot restarts, mode changes, or a fresh start is needed.
        """
        self.unmatched_buys.clear()
        self.unmatched_sells.clear()
        self.pnl_accumulator.clear()
        self.processed_trade_ids.clear()
        self.pnl_display_cache.clear()

    def copy(self) -> TradeMatcherState:
        """Create a deep copy of the state.

        Returns:
            A new TradeMatcherState with copied data.
        """
        return TradeMatcherState(
            unmatched_buys={k: list(v) for k, v in self.unmatched_buys.items()},
            unmatched_sells={k: list(v) for k, v in self.unmatched_sells.items()},
            pnl_accumulator=dict(self.pnl_accumulator),
            processed_trade_ids=set(self.processed_trade_ids),
            pnl_display_cache=dict(self.pnl_display_cache),
        )
