"""Trade statistics calculation for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_trader.backtesting.types import CompletedTrade, TradeOutcome

if TYPE_CHECKING:
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker
    from gpt_trader.features.brokerages.core.interfaces import Order


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics from a backtest run."""

    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    # Win/loss metrics
    win_rate: Decimal  # percentage
    loss_rate: Decimal  # percentage
    profit_factor: Decimal  # gross profit / gross loss

    # PnL metrics
    total_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    avg_profit_per_trade: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal

    # Position metrics
    avg_position_size_usd: Decimal
    max_position_size_usd: Decimal
    avg_leverage: Decimal
    max_leverage: Decimal

    # Execution quality
    avg_slippage_bps: Decimal
    total_fees_paid: Decimal
    limit_orders_filled: int
    limit_orders_cancelled: int
    limit_fill_rate: Decimal  # percentage

    # Timing
    avg_hold_time_minutes: Decimal
    max_hold_time_minutes: Decimal

    # Streak analysis
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int  # positive = wins, negative = losses


def calculate_trade_statistics(broker: SimulatedBroker) -> TradeStatistics:
    """
    Calculate comprehensive trade statistics from a SimulatedBroker.

    Args:
        broker: SimulatedBroker with completed trades

    Returns:
        TradeStatistics with all computed metrics
    """
    # Get completed trades for accurate PnL calculation
    completed_trades = broker.get_completed_trades()

    # Get all filled orders
    filled_orders = list(broker._filled_orders.values())
    cancelled_orders = list(broker._cancelled_orders.values())

    # Use completed trades if available, otherwise fall back to broker counters
    # (for backwards compatibility with tests that mock broker state directly)
    if completed_trades:
        total_trades = len(completed_trades)
        winning_trades = sum(1 for t in completed_trades if t.outcome == TradeOutcome.WIN)
        losing_trades = sum(1 for t in completed_trades if t.outcome == TradeOutcome.LOSS)
        breakeven_trades = sum(1 for t in completed_trades if t.outcome == TradeOutcome.BREAKEVEN)
    else:
        # Fall back to broker counters
        total_trades = broker._total_trades
        winning_trades = broker._winning_trades
        losing_trades = broker._losing_trades
        breakeven_trades = total_trades - winning_trades - losing_trades

    # Win/loss rates
    win_rate = (
        Decimal(winning_trades) / Decimal(total_trades) * 100 if total_trades > 0 else Decimal("0")
    )
    loss_rate = (
        Decimal(losing_trades) / Decimal(total_trades) * 100 if total_trades > 0 else Decimal("0")
    )

    # Calculate PnL metrics from completed trades (accurate) or fallback to legacy
    if completed_trades:
        pnl_data = _calculate_pnl_metrics_from_trades(completed_trades)
    else:
        pnl_data = _calculate_pnl_metrics(filled_orders)

    # Profit factor
    profit_factor = (
        abs(pnl_data["gross_profit"] / pnl_data["gross_loss"])
        if pnl_data["gross_loss"] != 0
        else Decimal("999.99")  # Cap at 999.99 for display
    )

    # Average metrics
    avg_profit = pnl_data["total_pnl"] / Decimal(total_trades) if total_trades > 0 else Decimal("0")
    avg_win = (
        pnl_data["gross_profit"] / Decimal(winning_trades) if winning_trades > 0 else Decimal("0")
    )
    avg_loss = (
        pnl_data["gross_loss"] / Decimal(losing_trades) if losing_trades > 0 else Decimal("0")
    )

    # Position metrics from completed trades or fallback to legacy
    if completed_trades:
        position_data = _calculate_position_metrics_from_trades(completed_trades, broker)
    else:
        position_data = _calculate_position_metrics(filled_orders, broker)

    # Execution quality
    limit_filled = sum(1 for o in filled_orders if o.type.value == "LIMIT")
    limit_cancelled = sum(1 for o in cancelled_orders if o.type.value == "LIMIT")
    total_limit_orders = limit_filled + limit_cancelled
    limit_fill_rate = (
        Decimal(limit_filled) / Decimal(total_limit_orders) * 100
        if total_limit_orders > 0
        else Decimal("100")  # 100% if no limit orders
    )

    avg_slippage = (
        broker._total_slippage_bps / Decimal(broker._total_trades)
        if broker._total_trades > 0
        else Decimal("0")
    )

    # Hold time from completed trades (accurate) or fallback to legacy
    if completed_trades:
        timing_data = _calculate_timing_metrics_from_trades(completed_trades)
    else:
        timing_data = _calculate_timing_metrics(filled_orders)

    # Streak analysis from completed trades (accurate) or fallback to legacy
    if completed_trades:
        streak_data = _calculate_streak_metrics_from_trades(completed_trades)
    else:
        streak_data = _calculate_streak_metrics(filled_orders)

    return TradeStatistics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=win_rate,
        loss_rate=loss_rate,
        profit_factor=profit_factor,
        total_pnl=pnl_data["total_pnl"],
        gross_profit=pnl_data["gross_profit"],
        gross_loss=pnl_data["gross_loss"],
        avg_profit_per_trade=avg_profit,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=pnl_data["largest_win"],
        largest_loss=pnl_data["largest_loss"],
        avg_position_size_usd=position_data["avg_size"],
        max_position_size_usd=position_data["max_size"],
        avg_leverage=position_data["avg_leverage"],
        max_leverage=position_data["max_leverage"],
        avg_slippage_bps=avg_slippage,
        total_fees_paid=broker._total_fees_paid,
        limit_orders_filled=limit_filled,
        limit_orders_cancelled=limit_cancelled,
        limit_fill_rate=limit_fill_rate,
        avg_hold_time_minutes=timing_data["avg_hold"],
        max_hold_time_minutes=timing_data["max_hold"],
        max_consecutive_wins=streak_data["max_wins"],
        max_consecutive_losses=streak_data["max_losses"],
        current_streak=streak_data["current"],
    )


def _calculate_pnl_metrics(orders: list[Order]) -> dict[str, Decimal]:
    """Calculate PnL metrics from filled orders."""
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    largest_win = Decimal("0")
    largest_loss = Decimal("0")

    # Group orders by symbol to calculate realized PnL
    # This is a simplified calculation - real PnL is tracked in positions
    for order in orders:
        if order.avg_fill_price is None:
            continue

        # Note: Actual PnL calculation requires position tracking
        # Here we estimate based on order flow
        # The real PnL comes from broker._credit_pnl calls during position closes

    # For accurate PnL, we use the broker's account state
    # This is placeholder - actual implementation pulls from position history
    return {
        "total_pnl": Decimal(str(gross_profit + gross_loss)),
        "gross_profit": Decimal(str(gross_profit)),
        "gross_loss": Decimal(str(gross_loss)),
        "largest_win": Decimal(str(largest_win)),
        "largest_loss": Decimal(str(largest_loss)),
    }


def _calculate_position_metrics(orders: list[Order], broker: SimulatedBroker) -> dict[str, Decimal]:
    """Calculate position size and leverage metrics."""
    if not orders:
        return {
            "avg_size": Decimal("0"),
            "max_size": Decimal("0"),
            "avg_leverage": Decimal("1"),
            "max_leverage": Decimal("1"),
        }

    position_sizes: list[Decimal] = []
    leverages: list[int] = []

    for order in orders:
        if order.avg_fill_price is None:
            continue

        notional = order.filled_quantity * order.avg_fill_price
        position_sizes.append(notional)

        # Try to get leverage from order metadata or position
        leverage = 1  # Default
        pos = broker.positions.get(order.symbol)
        if pos and pos.leverage:
            leverage = pos.leverage
        leverages.append(leverage)

    avg_size = (
        sum(position_sizes, Decimal("0")) / len(position_sizes) if position_sizes else Decimal("0")
    )
    max_size = max(position_sizes) if position_sizes else Decimal("0")
    avg_leverage = (
        Decimal(str(sum(leverages))) / Decimal(str(len(leverages))) if leverages else Decimal("1")
    )
    max_leverage = Decimal(str(max(leverages))) if leverages else Decimal("1")

    return {
        "avg_size": Decimal(str(avg_size)),
        "max_size": Decimal(str(max_size)),
        "avg_leverage": Decimal(str(avg_leverage)),
        "max_leverage": Decimal(str(max_leverage)),
    }


def _calculate_timing_metrics(orders: list[Order]) -> dict[str, Decimal]:
    """Calculate hold time metrics from orders."""
    hold_times: list[Decimal] = []

    # Pair entry and exit orders by symbol
    symbol_entries: dict[str, datetime] = {}

    for order in sorted(orders, key=lambda o: o.submitted_at or datetime.min):
        if order.submitted_at is None:
            continue

        symbol = order.symbol
        is_opening = order.side.value == "BUY"  # Simplified: buy = open

        if is_opening:
            symbol_entries[symbol] = order.submitted_at
        elif symbol in symbol_entries:
            entry_time = symbol_entries[symbol]
            exit_time = order.submitted_at
            hold_minutes = (exit_time - entry_time).total_seconds() / 60
            hold_times.append(Decimal(str(hold_minutes)))
            del symbol_entries[symbol]

    avg_hold = sum(hold_times, Decimal("0")) / len(hold_times) if hold_times else Decimal("0")
    max_hold = max(hold_times) if hold_times else Decimal("0")

    return {"avg_hold": Decimal(str(avg_hold)), "max_hold": Decimal(str(max_hold))}


def _calculate_streak_metrics(orders: list[Order]) -> dict[str, int]:
    """Calculate win/loss streak metrics."""
    # This requires PnL per trade which we don't fully track in orders
    # Placeholder implementation - returns zeros until position history tracking is added
    _ = orders  # Acknowledge unused parameter for now
    return {
        "max_wins": 0,
        "max_losses": 0,
        "current": 0,
    }


# =============================================================================
# New functions using CompletedTrade for accurate metrics
# =============================================================================


def _calculate_pnl_metrics_from_trades(trades: list[CompletedTrade]) -> dict[str, Decimal]:
    """
    Calculate accurate PnL metrics from completed trades.

    Args:
        trades: List of CompletedTrade objects

    Returns:
        Dictionary with PnL metrics:
        - total_pnl: Net PnL after fees
        - gross_profit: Sum of winning trades (before fees)
        - gross_loss: Sum of losing trades (before fees, negative)
        - largest_win: Largest winning net PnL
        - largest_loss: Largest losing net PnL (negative)
    """
    if not trades:
        return {
            "total_pnl": Decimal("0"),
            "gross_profit": Decimal("0"),
            "gross_loss": Decimal("0"),
            "largest_win": Decimal("0"),
            "largest_loss": Decimal("0"),
        }

    total_pnl = Decimal("0")
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    largest_win = Decimal("0")
    largest_loss = Decimal("0")

    for trade in trades:
        total_pnl += trade.net_pnl

        if trade.outcome == TradeOutcome.WIN:
            gross_profit += trade.realized_pnl
            if trade.net_pnl > largest_win:
                largest_win = trade.net_pnl
        elif trade.outcome == TradeOutcome.LOSS:
            gross_loss += trade.realized_pnl  # Will be negative
            if trade.net_pnl < largest_loss:
                largest_loss = trade.net_pnl

    return {
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
    }


def _calculate_position_metrics_from_trades(
    trades: list[CompletedTrade],
    broker: SimulatedBroker,
) -> dict[str, Decimal]:
    """
    Calculate position size and leverage metrics from completed trades.

    Args:
        trades: List of CompletedTrade objects
        broker: SimulatedBroker for additional context

    Returns:
        Dictionary with position metrics
    """
    if not trades:
        return {
            "avg_size": Decimal("0"),
            "max_size": Decimal("0"),
            "avg_leverage": Decimal("1"),
            "max_leverage": Decimal("1"),
        }

    position_sizes: list[Decimal] = []
    leverages: list[int] = []

    for trade in trades:
        # Calculate notional value (position size in USD)
        notional = trade.quantity * trade.entry_price
        position_sizes.append(notional)

        # Try to get leverage from current position or default to 1
        pos = broker.positions.get(trade.symbol)
        leverage = pos.leverage if pos and pos.leverage else 1
        leverages.append(leverage)

    avg_size = (
        sum(position_sizes, Decimal("0")) / len(position_sizes) if position_sizes else Decimal("0")
    )
    max_size = max(position_sizes) if position_sizes else Decimal("0")
    avg_leverage = (
        Decimal(str(sum(leverages))) / Decimal(str(len(leverages))) if leverages else Decimal("1")
    )
    max_leverage = Decimal(str(max(leverages))) if leverages else Decimal("1")

    return {
        "avg_size": avg_size,
        "max_size": max_size,
        "avg_leverage": avg_leverage,
        "max_leverage": max_leverage,
    }


def _calculate_timing_metrics_from_trades(trades: list[CompletedTrade]) -> dict[str, Decimal]:
    """
    Calculate hold time metrics from completed trades.

    Args:
        trades: List of CompletedTrade objects

    Returns:
        Dictionary with timing metrics:
        - avg_hold: Average hold time in minutes
        - max_hold: Maximum hold time in minutes
    """
    if not trades:
        return {"avg_hold": Decimal("0"), "max_hold": Decimal("0")}

    hold_times_minutes = [Decimal(str(trade.hold_time_seconds)) / Decimal("60") for trade in trades]

    if not hold_times_minutes:
        return {"avg_hold": Decimal("0"), "max_hold": Decimal("0")}

    avg_hold = sum(hold_times_minutes, Decimal("0")) / Decimal(len(hold_times_minutes))
    max_hold = max(hold_times_minutes)

    return {"avg_hold": avg_hold, "max_hold": max_hold}


def _calculate_streak_metrics_from_trades(trades: list[CompletedTrade]) -> dict[str, int]:
    """
    Calculate win/loss streak metrics from completed trades.

    Args:
        trades: List of CompletedTrade objects sorted by exit time

    Returns:
        Dictionary with streak metrics:
        - max_wins: Maximum consecutive winning trades
        - max_losses: Maximum consecutive losing trades
        - current: Current streak (positive = wins, negative = losses)
    """
    if not trades:
        return {"max_wins": 0, "max_losses": 0, "current": 0}

    # Sort trades by exit time for accurate streak calculation
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    max_wins = 0
    max_losses = 0
    current_win_streak = 0
    current_loss_streak = 0
    current_streak = 0

    for trade in sorted_trades:
        if trade.outcome == TradeOutcome.WIN:
            current_win_streak += 1
            current_loss_streak = 0
            current_streak = current_win_streak
            if current_win_streak > max_wins:
                max_wins = current_win_streak
        elif trade.outcome == TradeOutcome.LOSS:
            current_loss_streak += 1
            current_win_streak = 0
            current_streak = -current_loss_streak
            if current_loss_streak > max_losses:
                max_losses = current_loss_streak
        else:  # BREAKEVEN
            # Breakeven doesn't affect streaks but resets them
            current_win_streak = 0
            current_loss_streak = 0
            current_streak = 0

    return {
        "max_wins": max_wins,
        "max_losses": max_losses,
        "current": current_streak,
    }
