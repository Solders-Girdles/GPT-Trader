"""Trade matching and P&L calculation for TUI widgets.

This module provides trade matching functionality using FIFO matching
to pair buy/sell trades and calculate realized P&L.

The matching logic is now separated from state management:
- TradeMatcherState: Holds all mutable state (in state_management/)
- TradeMatcher: Orchestrates matching operations using state
- Pure functions: Perform the actual matching calculations
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from gpt_trader.tui.state_management import TradeMatcherState
from gpt_trader.tui.types import Trade


@dataclass
class TradeMatch:
    """Represents a matched BUY/SELL trade pair with calculated P&L."""

    entry_trade_id: str
    exit_trade_id: str
    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    realized_pnl: Decimal
    side: str  # "LONG" or "SHORT"


class TradeMatcher:
    """
    Matches BUY/SELL trade pairs and calculates realized P&L.

    Uses FIFO (First-In-First-Out) matching to pair trades:
    - For LONG positions: BUY is entry, SELL is exit
    - For SHORT positions: SELL is entry, BUY is exit

    Formula: realized_pnl = (exit_price - entry_price) * quantity - (entry_fee + exit_fee)

    State is now managed via TradeMatcherState dataclass, enabling:
    - Centralized state management
    - Proper reset handling on mode changes
    - State serialization/persistence if needed
    """

    def __init__(self, state: TradeMatcherState | None = None):
        """Initialize TradeMatcher with optional external state.

        Args:
            state: Optional TradeMatcherState instance. If not provided,
                creates a new internal state instance.
        """
        self._state = state if state is not None else TradeMatcherState()

    @property
    def state(self) -> TradeMatcherState:
        """Access the underlying state."""
        return self._state

    @state.setter
    def state(self, value: TradeMatcherState) -> None:
        """Set the state (e.g., from central state management)."""
        self._state = value

    def process_trades(self, trades: list[Trade]) -> dict[str, str | None]:
        """
        Process a list of trades and return P&L display values.

        Uses incremental processing - only new trades (not in processed_trade_ids)
        are matched. This changes from O(n) to O(new_trades) complexity.

        Args:
            trades: List of Trade objects (should be in chronological order)

        Returns:
            Dict mapping trade_id to P&L display string (e.g., "+150.25", "-42.10", "N/A")
        """
        state = self._state

        # Identify new trades that haven't been processed yet
        new_trades = [trade for trade in trades if trade.trade_id not in state.processed_trade_ids]

        # Track which trade IDs had P&L changes (for cache updates)
        affected_trade_ids: set[str] = set()

        # Process only new trades (O(new) instead of O(all))
        for trade in new_trades:
            # Get snapshot of P&L before matching
            before_pnl_keys = set(state.pnl_accumulator.keys())

            # Match the trade
            self._match_trade(trade)
            state.processed_trade_ids.add(trade.trade_id)

            # Get snapshot of P&L after matching - any new or changed entries were affected
            after_pnl_keys = set(state.pnl_accumulator.keys())
            affected_trade_ids.update(after_pnl_keys)

            # Also check for changes in existing entries
            # (when a new trade matches an old one, the old trade's P&L changes)
            for trade_id in before_pnl_keys:
                affected_trade_ids.add(trade_id)

        # Update display cache for ALL affected trades (including matched old trades)
        for trade_id in affected_trade_ids:
            pnl_str = self._get_pnl_display(trade_id)
            state.pnl_display_cache[trade_id] = pnl_str

        # Return display map for ALL trades (uses cached values + updated calculations)
        return {
            trade.trade_id: state.pnl_display_cache.get(trade.trade_id, "N/A") for trade in trades
        }

    def _match_trade(self, trade: Trade) -> None:
        """Match a single trade against unmatched trades."""
        symbol = trade.symbol
        state = self._state

        try:
            quantity = Decimal(str(trade.quantity))
            price = Decimal(str(trade.price))
            fee = Decimal(str(trade.fee))
        except (ValueError, InvalidOperation):
            # Can't parse trade - skip matching
            return

        # Initialize symbol queues if needed
        if symbol not in state.unmatched_buys:
            state.unmatched_buys[symbol] = []
        if symbol not in state.unmatched_sells:
            state.unmatched_sells[symbol] = []

        if trade.side == "BUY":
            self._match_buy(trade, quantity, price, fee)
        elif trade.side == "SELL":
            self._match_sell(trade, quantity, price, fee)

    def _match_buy(self, trade: Trade, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        """Match a BUY trade (could close SHORT or open LONG)."""
        symbol = trade.symbol
        state = self._state
        remaining_quantity = quantity

        # Check if this BUY closes any SHORT positions (SELL entries)
        unmatched_sells = state.unmatched_sells[symbol]

        i = 0
        while i < len(unmatched_sells) and remaining_quantity > 0:
            sell_trade, sell_quantity = unmatched_sells[i]

            # Calculate match quantity
            match_quantity = min(remaining_quantity, sell_quantity)

            # Calculate P&L for SHORT position
            # SHORT P&L: (entry_price - exit_price) * quantity - fees
            sell_price = Decimal(str(sell_trade.price))
            sell_fee = Decimal(str(sell_trade.fee))

            # Proportional fees
            entry_fee_portion = (sell_fee * match_quantity) / sell_quantity
            exit_fee_portion = (fee * match_quantity) / quantity

            pnl = (sell_price - price) * match_quantity - (entry_fee_portion + exit_fee_portion)

            # Accumulate P&L for both entry and exit trades
            state.pnl_accumulator[sell_trade.trade_id] = (
                state.pnl_accumulator.get(sell_trade.trade_id, Decimal("0")) + pnl
            )
            state.pnl_accumulator[trade.trade_id] = (
                state.pnl_accumulator.get(trade.trade_id, Decimal("0")) + pnl
            )

            # Update remaining quantities
            remaining_quantity -= match_quantity
            sell_quantity -= match_quantity

            if sell_quantity == 0:
                # Fully matched - remove from queue
                unmatched_sells.pop(i)
            else:
                # Partially matched - update queue
                unmatched_sells[i] = (sell_trade, sell_quantity)
                i += 1

        # If quantity remains, add as unmatched BUY (opens LONG)
        if remaining_quantity > 0:
            state.unmatched_buys[symbol].append((trade, remaining_quantity))

    def _match_sell(self, trade: Trade, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        """Match a SELL trade (could close LONG or open SHORT)."""
        symbol = trade.symbol
        state = self._state
        remaining_quantity = quantity

        # Check if this SELL closes any LONG positions (BUY entries)
        unmatched_buys = state.unmatched_buys[symbol]

        i = 0
        while i < len(unmatched_buys) and remaining_quantity > 0:
            buy_trade, buy_quantity = unmatched_buys[i]

            # Calculate match quantity
            match_quantity = min(remaining_quantity, buy_quantity)

            # Calculate P&L for LONG position
            # LONG P&L: (exit_price - entry_price) * quantity - fees
            buy_price = Decimal(str(buy_trade.price))
            buy_fee = Decimal(str(buy_trade.fee))

            # Proportional fees
            entry_fee_portion = (buy_fee * match_quantity) / buy_quantity
            exit_fee_portion = (fee * match_quantity) / quantity

            pnl = (price - buy_price) * match_quantity - (entry_fee_portion + exit_fee_portion)

            # Accumulate P&L for both entry and exit trades
            state.pnl_accumulator[buy_trade.trade_id] = (
                state.pnl_accumulator.get(buy_trade.trade_id, Decimal("0")) + pnl
            )
            state.pnl_accumulator[trade.trade_id] = (
                state.pnl_accumulator.get(trade.trade_id, Decimal("0")) + pnl
            )

            # Update remaining quantities
            remaining_quantity -= match_quantity
            buy_quantity -= match_quantity

            if buy_quantity == 0:
                # Fully matched - remove from queue
                unmatched_buys.pop(i)
            else:
                # Partially matched - update queue
                unmatched_buys[i] = (buy_trade, buy_quantity)
                i += 1

        # If quantity remains, add as unmatched SELL (opens SHORT)
        if remaining_quantity > 0:
            state.unmatched_sells[symbol].append((trade, remaining_quantity))

    def reset(self) -> None:
        """
        Reset all state (call on bot restart or mode switch).

        Clears all internal state to start fresh P&L tracking.
        Delegates to the TradeMatcherState.reset() method.
        """
        self._state.reset()

    def _get_pnl_display(self, trade_id: str) -> str | None:
        """Get P&L display string for a trade."""
        if trade_id not in self._state.pnl_accumulator:
            return "N/A"

        pnl_value = float(self._state.pnl_accumulator[trade_id])

        # Format with sign prefix
        if pnl_value >= 0:
            return f"+{pnl_value:.2f}"
        else:
            return f"{pnl_value:.2f}"  # Already has negative sign
