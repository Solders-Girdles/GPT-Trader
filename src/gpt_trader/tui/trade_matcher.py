"""Trade matching and P&L calculation for TUI widgets."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

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
    """

    def __init__(self):
        # Store unmatched trades by symbol
        # Key: symbol, Value: list of (trade, remaining_quantity)
        self._unmatched_buys: dict[str, list[tuple[Trade, Decimal]]] = {}
        self._unmatched_sells: dict[str, list[tuple[Trade, Decimal]]] = {}

        # Store accumulated P&L per trade_id (to handle partial matches)
        self._pnl_accumulator: dict[str, Decimal] = {}

        # NEW: Track processed trades for incremental updates (Phase 6)
        self._processed_trade_ids: set[str] = set()
        self._pnl_display_cache: dict[str, Optional[str]] = {}

    def process_trades(self, trades: list[Trade]) -> dict[str, Optional[str]]:
        """
        Process a list of trades and return P&L display values.

        Uses incremental processing - only new trades (not in _processed_trade_ids)
        are matched. This changes from O(n) to O(new_trades) complexity.

        Args:
            trades: List of Trade objects (should be in chronological order)

        Returns:
            Dict mapping trade_id to P&L display string (e.g., "+150.25", "-42.10", "N/A")
        """
        # Identify new trades that haven't been processed yet
        new_trades = [
            trade for trade in trades if trade.trade_id not in self._processed_trade_ids
        ]

        # Process only new trades (O(new) instead of O(all))
        for trade in new_trades:
            self._match_trade(trade)
            self._processed_trade_ids.add(trade.trade_id)

            # Update display cache for this trade
            pnl_str = self._get_pnl_display(trade.trade_id)
            self._pnl_display_cache[trade.trade_id] = pnl_str

        # Return display map for ALL trades (uses cached values + new calculations)
        return {
            trade.trade_id: self._pnl_display_cache.get(trade.trade_id, "N/A")
            for trade in trades
        }

    def _match_trade(self, trade: Trade) -> None:
        """Match a single trade against unmatched trades."""
        symbol = trade.symbol

        try:
            quantity = Decimal(str(trade.quantity))
            price = Decimal(str(trade.price))
            fee = Decimal(str(trade.fee))
        except (ValueError, InvalidOperation):
            # Can't parse trade - skip matching
            return

        # Initialize symbol queues if needed
        if symbol not in self._unmatched_buys:
            self._unmatched_buys[symbol] = []
        if symbol not in self._unmatched_sells:
            self._unmatched_sells[symbol] = []

        if trade.side == "BUY":
            self._match_buy(trade, quantity, price, fee)
        elif trade.side == "SELL":
            self._match_sell(trade, quantity, price, fee)

    def _match_buy(self, trade: Trade, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        """Match a BUY trade (could close SHORT or open LONG)."""
        symbol = trade.symbol
        remaining_qty = quantity

        # Check if this BUY closes any SHORT positions (SELL entries)
        unmatched_sells = self._unmatched_sells[symbol]

        i = 0
        while i < len(unmatched_sells) and remaining_qty > 0:
            sell_trade, sell_qty = unmatched_sells[i]

            # Calculate match quantity
            match_qty = min(remaining_qty, sell_qty)

            # Calculate P&L for SHORT position
            # SHORT P&L: (entry_price - exit_price) * quantity - fees
            sell_price = Decimal(str(sell_trade.price))
            sell_fee = Decimal(str(sell_trade.fee))

            # Proportional fees
            entry_fee_portion = (sell_fee * match_qty) / sell_qty
            exit_fee_portion = (fee * match_qty) / quantity

            pnl = (sell_price - price) * match_qty - (entry_fee_portion + exit_fee_portion)

            # Accumulate P&L for both entry and exit trades
            self._pnl_accumulator[sell_trade.trade_id] = (
                self._pnl_accumulator.get(sell_trade.trade_id, Decimal("0")) + pnl
            )
            self._pnl_accumulator[trade.trade_id] = (
                self._pnl_accumulator.get(trade.trade_id, Decimal("0")) + pnl
            )

            # Update remaining quantities
            remaining_qty -= match_qty
            sell_qty -= match_qty

            if sell_qty == 0:
                # Fully matched - remove from queue
                unmatched_sells.pop(i)
            else:
                # Partially matched - update queue
                unmatched_sells[i] = (sell_trade, sell_qty)
                i += 1

        # If quantity remains, add as unmatched BUY (opens LONG)
        if remaining_qty > 0:
            self._unmatched_buys[symbol].append((trade, remaining_qty))

    def _match_sell(self, trade: Trade, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        """Match a SELL trade (could close LONG or open SHORT)."""
        symbol = trade.symbol
        remaining_qty = quantity

        # Check if this SELL closes any LONG positions (BUY entries)
        unmatched_buys = self._unmatched_buys[symbol]

        i = 0
        while i < len(unmatched_buys) and remaining_qty > 0:
            buy_trade, buy_qty = unmatched_buys[i]

            # Calculate match quantity
            match_qty = min(remaining_qty, buy_qty)

            # Calculate P&L for LONG position
            # LONG P&L: (exit_price - entry_price) * quantity - fees
            buy_price = Decimal(str(buy_trade.price))
            buy_fee = Decimal(str(buy_trade.fee))

            # Proportional fees
            entry_fee_portion = (buy_fee * match_qty) / buy_qty
            exit_fee_portion = (fee * match_qty) / quantity

            pnl = (price - buy_price) * match_qty - (entry_fee_portion + exit_fee_portion)

            # Accumulate P&L for both entry and exit trades
            self._pnl_accumulator[buy_trade.trade_id] = (
                self._pnl_accumulator.get(buy_trade.trade_id, Decimal("0")) + pnl
            )
            self._pnl_accumulator[trade.trade_id] = (
                self._pnl_accumulator.get(trade.trade_id, Decimal("0")) + pnl
            )

            # Update remaining quantities
            remaining_qty -= match_qty
            buy_qty -= match_qty

            if buy_qty == 0:
                # Fully matched - remove from queue
                unmatched_buys.pop(i)
            else:
                # Partially matched - update queue
                unmatched_buys[i] = (buy_trade, buy_qty)
                i += 1

        # If quantity remains, add as unmatched SELL (opens SHORT)
        if remaining_qty > 0:
            self._unmatched_sells[symbol].append((trade, remaining_qty))

    def reset(self) -> None:
        """
        Reset all state (call on bot restart or mode switch).

        Clears all internal state to start fresh P&L tracking.
        """
        self._unmatched_buys.clear()
        self._unmatched_sells.clear()
        self._pnl_accumulator.clear()
        self._processed_trade_ids.clear()
        self._pnl_display_cache.clear()

    def _get_pnl_display(self, trade_id: str) -> Optional[str]:
        """Get P&L display string for a trade."""
        if trade_id not in self._pnl_accumulator:
            return "N/A"

        pnl_value = float(self._pnl_accumulator[trade_id])

        # Format with sign prefix
        if pnl_value >= 0:
            return f"+{pnl_value:.2f}"
        else:
            return f"{pnl_value:.2f}"  # Already has negative sign
