"""
PnL and Funding tracking for Week 3.

Tracks realized/unrealized PnL, funding accruals, and daily metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """
    Tracks position state and PnL calculations.

    Maintains realized/unrealized PnL, funding paid/received,
    and position history for a single symbol.
    """

    symbol: str
    side: str | None = None  # 'long' or 'short'
    quantity: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")

    # PnL tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Funding tracking
    funding_paid: Decimal = Decimal("0")  # Total funding paid (positive = paid out)
    last_funding_time: datetime | None = None

    # Statistics
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")

    def update_position(
        self, side: str, quantity: Decimal, price: Decimal, is_reduce: bool = False
    ) -> dict[str, Decimal]:
        """
        Update position with new trade.

        Args:
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Execution price
            is_reduce: Whether this reduces position

        Returns:
            Dict with realized_pnl from this trade
        """
        result = {"realized_pnl": Decimal("0")}

        if self.quantity == 0:
            # Opening new position
            self.side = "long" if side == "buy" else "short"
            self.quantity = quantity
            self.avg_entry_price = price
            self.trades_count += 1

        elif (
            is_reduce
            or (self.side == "long" and side == "sell")
            or (self.side == "short" and side == "buy")
        ):
            # Reducing or closing position
            close_quantity = min(quantity, abs(self.quantity))

            # Calculate realized PnL
            if self.side == "long":
                pnl = (price - self.avg_entry_price) * close_quantity
            else:  # short
                pnl = (self.avg_entry_price - price) * close_quantity

            self.realized_pnl += pnl
            result["realized_pnl"] = pnl

            # Update win/loss stats
            if pnl > 0:
                self.winning_trades += 1
            elif pnl < 0:
                self.losing_trades += 1

            # Update position
            self.quantity -= close_quantity
            if self.quantity == 0:
                self.side = None
                self.avg_entry_price = Decimal("0")

            # Check if flipping position
            if quantity > close_quantity:
                remaining = quantity - close_quantity
                self.side = "long" if side == "buy" else "short"
                self.quantity = remaining
                self.avg_entry_price = price
                self.trades_count += 1

        else:
            # Adding to position
            total_quantity = self.quantity + quantity
            # Weighted average entry
            self.avg_entry_price = (
                self.avg_entry_price * self.quantity + price * quantity
            ) / total_quantity
            self.quantity = total_quantity

        return result

    def update_mark(self, mark_price: Decimal) -> Decimal:
        """
        Update unrealized PnL with current mark price.

        Args:
            mark_price: Current mark price

        Returns:
            Current unrealized PnL
        """
        if self.quantity == 0:
            self.unrealized_pnl = Decimal("0")
        elif self.side == "long":
            self.unrealized_pnl = (mark_price - self.avg_entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.avg_entry_price - mark_price) * abs(self.quantity)

        # Update drawdown
        total_pnl = self.realized_pnl + self.unrealized_pnl
        if total_pnl > self.peak_equity:
            self.peak_equity = total_pnl

        drawdown = self.peak_equity - total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        return self.unrealized_pnl

    def get_metrics(self) -> dict[str, float | int | str | None | bool]:
        """Get position metrics."""
        win_rate = Decimal("0")
        if self.trades_count > 0:
            win_rate = Decimal(self.winning_trades) / Decimal(self.trades_count)

        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "avg_entry": float(self.avg_entry_price),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.realized_pnl + self.unrealized_pnl),
            "funding_paid": float(self.funding_paid),
            "trades": self.trades_count,
            "win_rate": float(win_rate),
            "max_drawdown": float(self.max_drawdown),
        }


@dataclass
class FundingCalculator:
    """
    Calculates and tracks funding payments for perpetual contracts.

    Funding conventions:
    - Positive rate: longs pay shorts
    - Negative rate: shorts pay longs
    - Payments occur at scheduled intervals (typically 8h)
    """

    funding_interval_hours: int = 8  # Typical for perps

    def calculate_funding(
        self, position_size: Decimal, side: str | None, mark_price: Decimal, funding_rate: Decimal
    ) -> Decimal:
        """
        Calculate funding payment/receipt.

        Args:
            position_size: Position size (quantity)
            side: 'long' or 'short'
            mark_price: Current mark price
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)

        Returns:
            Funding amount (positive = received, negative = paid)
        """
        if position_size == 0 or side is None:
            return Decimal("0")

        # Calculate notional value
        notional = position_size * mark_price

        # Apply funding rate
        funding = notional * funding_rate

        # Adjust sign based on side
        # Positive rate: longs pay (negative), shorts receive (positive)
        # Negative rate: shorts pay (negative), longs receive (positive)
        if side == "long":
            return -funding  # Longs pay when rate positive
        else:  # short
            return funding  # Shorts receive when rate positive

    def is_funding_due(
        self,
        last_funding_time: datetime | None,
        next_funding_time: datetime | None,
        current_time: datetime | None = None,
    ) -> bool:
        """
        Check if funding payment is due.

        Args:
            last_funding_time: Last time funding was paid
            next_funding_time: Next scheduled funding time
            current_time: Current time (defaults to now)

        Returns:
            True if funding is due
        """
        if current_time is None:
            current_time = datetime.now()

        # If we have next funding time, use it
        if next_funding_time:
            return current_time >= next_funding_time

        # Otherwise check interval
        if last_funding_time:
            time_since = current_time - last_funding_time
            return time_since >= timedelta(hours=self.funding_interval_hours)

        # First funding
        return True

    def accrue_if_due(
        self,
        position_state: PositionState,
        mark_price: Decimal,
        funding_rate: Decimal,
        next_funding_time: datetime | None = None,
    ) -> Decimal | None:
        """
        Accrue funding if due for position.

        Args:
            position_state: Current position state
            mark_price: Current mark price
            funding_rate: Current funding rate
            next_funding_time: Next scheduled funding

        Returns:
            Funding amount if accrued, None otherwise
        """
        if position_state.quantity == 0:
            return None

        if not self.is_funding_due(position_state.last_funding_time, next_funding_time):
            return None

        # Calculate funding
        funding = self.calculate_funding(
            position_state.quantity, position_state.side, mark_price, funding_rate
        )

        # Update position state
        position_state.funding_paid -= funding  # Negative funding = we receive
        position_state.last_funding_time = datetime.now()

        logger.info(
            f"Funding accrued for {position_state.symbol}: "
            f"{funding:+.4f} ({position_state.side} {position_state.quantity} @ {mark_price})"
        )

        return funding


class PnLTracker:
    """
    Comprehensive PnL and metrics tracker.

    Manages position states, funding accruals, and generates
    daily performance metrics.
    """

    def __init__(self) -> None:
        """Initialize PnL tracker."""
        self.positions: dict[str, PositionState] = {}
        self.funding_calculation = FundingCalculator()

        # Daily metrics
        self.daily_start_equity: Decimal | None = None
        self.daily_start_time: datetime | None = None

    def get_or_create_position(self, symbol: str) -> PositionState:
        """Get or create position state for symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        return self.positions[symbol]

    def update_position(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal, is_reduce: bool = False
    ) -> dict[str, Decimal]:
        """Update position with trade."""
        position = self.get_or_create_position(symbol)
        return position.update_position(side, quantity, price, is_reduce)

    def update_marks(self, mark_prices: dict[str, Decimal]) -> dict[str, Decimal]:
        """Update all positions with current marks."""
        unrealized = {}
        for symbol, mark in mark_prices.items():
            if symbol in self.positions:
                unrealized[symbol] = self.positions[symbol].update_mark(mark)
        return unrealized

    def accrue_funding(
        self,
        symbol: str,
        mark_price: Decimal,
        funding_rate: Decimal,
        next_funding_time: datetime | None = None,
    ) -> Decimal | None:
        """Accrue funding for symbol if due."""
        position = self.get_or_create_position(symbol)
        return self.funding_calculation.accrue_if_due(
            position, mark_price, funding_rate, next_funding_time
        )

    def get_total_pnl(self) -> dict[str, Decimal]:
        """Get total PnL across all positions."""
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")
        total_funding = Decimal("0")

        for position in self.positions.values():
            total_realized += position.realized_pnl
            total_unrealized += position.unrealized_pnl
            total_funding += position.funding_paid

        return {
            "realized": total_realized,
            "unrealized": total_unrealized,
            "total": total_realized + total_unrealized,
            "funding": total_funding,
        }

    def generate_daily_metrics(self, current_equity: Decimal) -> dict[str, float | int | str]:
        """
        Generate daily performance metrics.

        Args:
            current_equity: Current account equity

        Returns:
            Daily performance snapshot
        """
        now = datetime.now()

        # Calculate current PnL snapshot first
        pnl = self.get_total_pnl()

        # Initialize daily tracking (first call each day)
        if self.daily_start_time is None or (now - self.daily_start_time) >= timedelta(days=1):
            # Infer start-of-day equity from current equity minus PnL accrued so far
            self.daily_start_equity = current_equity - (pnl["realized"] + pnl["unrealized"])
            self.daily_start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_return = Decimal("0")
        if self.daily_start_equity and self.daily_start_equity > 0:
            daily_return = (current_equity - self.daily_start_equity) / self.daily_start_equity

        # Aggregate position metrics
        total_trades = sum(p.trades_count for p in self.positions.values())
        winning_trades = sum(p.winning_trades for p in self.positions.values())

        win_rate = Decimal("0")
        if total_trades > 0:
            win_rate = Decimal(winning_trades) / Decimal(total_trades)

        max_drawdown = max((p.max_drawdown for p in self.positions.values()), default=Decimal("0"))

        # Calculate Sharpe (simplified - would need returns history)
        sharpe = Decimal("0")
        if daily_return != 0:
            # Simplified: assume 16% annual vol for crypto
            daily_vol = Decimal("0.01")  # ~1% daily
            sharpe = (daily_return / daily_vol) * Decimal("15.87")  # sqrt(252)

        return {
            "timestamp": now.isoformat(),
            "equity": float(current_equity),
            "daily_return": float(daily_return),
            "total_pnl": float(pnl["total"]),
            "realized_pnl": float(pnl["realized"]),
            "unrealized_pnl": float(pnl["unrealized"]),
            "funding_paid": float(pnl["funding"]),
            "trades": total_trades,
            "win_rate": float(win_rate),
            "max_drawdown": float(max_drawdown),
            "sharpe": float(sharpe),
            "positions": len([p for p in self.positions.values() if p.quantity != 0]),
        }

    def get_position_metrics(self) -> list[dict[str, float | int | str | None | bool]]:
        """Get metrics for all positions."""
        return [p.get_metrics() for p in self.positions.values()]
