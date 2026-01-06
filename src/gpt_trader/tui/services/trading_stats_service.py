"""
Trading Statistics Service.

Computes trading performance metrics from trade history with time window
support and sample size tracking for informed decision-making.
"""

# naming: allow - qty is standard trading abbreviation for quantity

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_trader.tui.types import Trade, TradingStats
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


# Available time windows (minutes, 0 = all session)
TIME_WINDOWS = [
    (0, "All Session"),
    (5, "Last 5 min"),
    (15, "Last 15 min"),
    (30, "Last 30 min"),
    (60, "Last 1 hour"),
]


@dataclass
class MatchedTrade:
    """A matched entry/exit trade pair with realized P&L.

    For simplicity, this uses a FIFO matching approach where sells
    are matched against the oldest buys.
    """

    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    side: str  # "long" or "short"
    entry_time: float
    exit_time: float
    pnl: Decimal
    fees: Decimal

    @property
    def is_winner(self) -> bool:
        """Check if this trade was profitable."""
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        """Check if this trade was a loss."""
        return self.pnl < 0


class TradingStatsService:
    """Service for computing trading statistics.

    Tracks trade history and computes performance metrics with
    support for time window filtering.
    """

    def __init__(self) -> None:
        """Initialize the trading stats service."""
        self._session_start = time.time()
        self._current_window_index = 0  # Index into TIME_WINDOWS
        self._matched_trades: list[MatchedTrade] = []

    @property
    def current_window(self) -> tuple[int, str]:
        """Get current time window (minutes, label)."""
        return TIME_WINDOWS[self._current_window_index]

    def cycle_window(self) -> tuple[int, str]:
        """Cycle to next time window and return it."""
        self._current_window_index = (self._current_window_index + 1) % len(TIME_WINDOWS)
        window = TIME_WINDOWS[self._current_window_index]
        logger.debug("Trading stats window changed to: %s", window[1])
        return window

    def reset_window(self) -> tuple[int, str]:
        """Reset to 'All Session' window."""
        self._current_window_index = 0
        return TIME_WINDOWS[0]

    def compute_stats(
        self,
        trades: list[Trade],
        window_minutes: int | None = None,
    ) -> TradingStats:
        """Compute trading statistics from trade history.

        Args:
            trades: List of Trade objects from trade history.
            window_minutes: Time window in minutes (0 or None = all session).

        Returns:
            TradingStats with computed metrics and sample sizes.
        """
        if window_minutes is None:
            window_minutes, window_label = self.current_window
        else:
            # Find matching label
            window_label = next(
                (label for mins, label in TIME_WINDOWS if mins == window_minutes),
                f"Last {window_minutes} min" if window_minutes > 0 else "All Session",
            )

        stats = TradingStats(
            window_minutes=window_minutes,
            window_label=window_label,
        )

        if not trades:
            return stats

        # Filter trades by time window
        now = time.time()
        if window_minutes > 0:
            cutoff = now - (window_minutes * 60)
            filtered_trades = [t for t in trades if self._parse_trade_time(t) >= cutoff]
        else:
            filtered_trades = trades

        if not filtered_trades:
            return stats

        # Match trades into entry/exit pairs
        matched = self._match_trades(filtered_trades)

        if not matched:
            # No matched pairs yet - show raw trade count
            stats.total_trades = len(filtered_trades)
            stats.avg_trade_size = sum(t.quantity for t in filtered_trades) / len(filtered_trades)
            return stats

        # Compute statistics from matched trades
        stats.total_trades = len(matched)

        wins = [t for t in matched if t.is_winner]
        losses = [t for t in matched if t.is_loser]
        breakeven = [t for t in matched if t.pnl == 0]

        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.break_even_trades = len(breakeven)

        # Win rate
        if stats.total_trades > 0:
            stats.win_rate = stats.winning_trades / stats.total_trades

        # Gross profit/loss
        stats.gross_profit = sum(t.pnl for t in wins) if wins else Decimal("0")
        stats.gross_loss = abs(sum(t.pnl for t in losses)) if losses else Decimal("0")
        stats.total_pnl = stats.gross_profit - stats.gross_loss

        # Average win/loss
        if wins:
            stats.avg_win = stats.gross_profit / len(wins)
        if losses:
            stats.avg_loss = stats.gross_loss / len(losses)

        # Profit factor
        if stats.gross_loss > 0:
            stats.profit_factor = float(stats.gross_profit / stats.gross_loss)
        elif stats.gross_profit > 0:
            stats.profit_factor = float("inf")

        # Average trade
        if stats.total_trades > 0:
            stats.avg_trade_pnl = stats.total_pnl / stats.total_trades
            stats.avg_trade_size = sum(t.quantity for t in matched) / stats.total_trades

        return stats

    def compute_from_state(self, state: TuiState) -> TradingStats:
        """Compute stats from TuiState's trade history.

        Args:
            state: Current TUI state with trade data.

        Returns:
            TradingStats with computed metrics.
        """
        trades = state.trade_data.trades if state.trade_data else []
        return self.compute_stats(trades)

    def _parse_trade_time(self, trade: Trade) -> float:
        """Parse trade timestamp to epoch seconds.

        Args:
            trade: Trade object with time field.

        Returns:
            Epoch timestamp (float).
        """
        if not trade.time:
            return 0.0

        # Try parsing ISO format (2024-01-15T10:30:00Z)
        from datetime import datetime

        try:
            # Handle various ISO formats
            time_str = trade.time.rstrip("Z")
            if "T" in time_str:
                if "." in time_str:
                    # With microseconds
                    dt = datetime.fromisoformat(time_str.split(".")[0])
                else:
                    dt = datetime.fromisoformat(time_str)
                return dt.timestamp()
        except (ValueError, AttributeError):
            pass

        # Try as epoch float
        try:
            return float(trade.time)
        except (ValueError, TypeError):
            return 0.0

    def _match_trades(self, trades: list[Trade]) -> list[MatchedTrade]:
        """Match buy/sell trades into completed round-trips using FIFO.

        This is a simplified FIFO matcher that pairs sells with oldest buys.
        For more complex scenarios (partial fills, multiple symbols), a
        more sophisticated matcher would be needed.

        Args:
            trades: List of trades to match.

        Returns:
            List of matched trade pairs with P&L.
        """
        matched: list[MatchedTrade] = []

        # Group by symbol
        by_symbol: dict[str, list[Trade]] = {}
        for t in trades:
            by_symbol.setdefault(t.symbol, []).append(t)

        for symbol, symbol_trades in by_symbol.items():
            # Sort by time
            sorted_trades = sorted(symbol_trades, key=lambda t: self._parse_trade_time(t))

            # Track open positions (FIFO queue)
            open_buys: list[tuple[Trade, Decimal]] = []  # (trade, remaining_qty)

            for trade in sorted_trades:
                side = trade.side.upper()
                qty = trade.quantity

                if side == "BUY":
                    open_buys.append((trade, qty))
                elif side == "SELL" and open_buys:
                    # Match against oldest buys (FIFO)
                    remaining_sell = qty

                    while remaining_sell > 0 and open_buys:
                        buy_trade, buy_remaining = open_buys[0]

                        match_qty = min(remaining_sell, buy_remaining)

                        # Calculate P&L for this match
                        pnl = (trade.price - buy_trade.price) * match_qty
                        fees = (buy_trade.fee + trade.fee) * (match_qty / trade.quantity)

                        matched.append(
                            MatchedTrade(
                                symbol=symbol,
                                entry_price=buy_trade.price,
                                exit_price=trade.price,
                                quantity=match_qty,
                                side="long",
                                entry_time=self._parse_trade_time(buy_trade),
                                exit_time=self._parse_trade_time(trade),
                                pnl=pnl - fees,
                                fees=fees,
                            )
                        )

                        remaining_sell -= match_qty
                        buy_remaining -= match_qty

                        if buy_remaining <= 0:
                            open_buys.pop(0)
                        else:
                            open_buys[0] = (buy_trade, buy_remaining)

        return matched


# Global singleton
_trading_stats_service: TradingStatsService | None = None


def get_trading_stats_service() -> TradingStatsService:
    """Get or create the global trading stats service."""
    global _trading_stats_service
    if _trading_stats_service is None:
        _trading_stats_service = TradingStatsService()
    return _trading_stats_service


def clear_trading_stats_service() -> None:
    """Clear the global trading stats service."""
    global _trading_stats_service
    _trading_stats_service = None


__all__ = [
    "MatchedTrade",
    "TIME_WINDOWS",
    "TradingStats",
    "TradingStatsService",
    "clear_trading_stats_service",
    "get_trading_stats_service",
]
