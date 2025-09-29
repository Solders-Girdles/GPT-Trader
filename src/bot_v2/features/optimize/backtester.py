"""
Local backtesting for optimization.

Complete isolation - no external dependencies.
"""

from typing import Any

import numpy as np
import pandas as pd

from .strategies import create_local_strategy
from .types import BacktestMetrics


def run_backtest_local(
    strategy: str,
    data: pd.DataFrame,
    params: dict[str, Any],
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> BacktestMetrics:
    """
    Run a backtest with given parameters.

    Args:
        strategy: Strategy name
        data: Historical OHLC data
        params: Strategy parameters
        commission: Commission rate
        slippage: Slippage rate

    Returns:
        Backtest metrics
    """
    # Create strategy instance
    strategy_instance = create_local_strategy(strategy, **params)

    # Generate signals
    signals = strategy_instance.generate_signals(data)

    # Simulate trades
    trades, equity_curve = simulate_trades(signals, data, commission, slippage)

    # Calculate metrics
    metrics = calculate_metrics(trades, equity_curve)

    return metrics


def simulate_trades(
    signals: pd.Series,
    data: pd.DataFrame,
    commission: float,
    slippage: float,
    initial_capital: float = 10000,
) -> tuple[list[dict], pd.Series]:
    """
    Simulate trade execution.

    Args:
        signals: Trading signals
        data: Market data
        commission: Commission rate
        slippage: Slippage rate
        initial_capital: Starting capital

    Returns:
        (List of trades, Equity curve)
    """
    trades = []
    cash = initial_capital
    position = 0
    equity_curve = []

    for i, (date, signal) in enumerate(signals.items()):
        if date not in data.index:
            continue

        price = data.loc[date, "close"]

        # Update equity
        equity = cash + (position * price)
        equity_curve.append(equity)

        # Process signals
        if signal == 1 and position == 0:
            # Buy
            buy_price = price * (1 + slippage)
            shares = int((cash * 0.95) / buy_price)

            if shares > 0:
                cost = shares * buy_price * (1 + commission)
                if cost <= cash:
                    cash -= cost
                    position = shares

                    trades.append(
                        {
                            "date": date,
                            "type": "buy",
                            "price": buy_price,
                            "shares": shares,
                            "value": cost,
                        }
                    )

        elif signal == -1 and position > 0:
            # Sell
            sell_price = price * (1 - slippage)
            proceeds = position * sell_price * (1 - commission)
            cash += proceeds

            # Calculate P&L
            if trades and trades[-1]["type"] == "buy":
                entry_price = trades[-1]["price"]
                pnl = (sell_price - entry_price) * position
                pnl_pct = pnl / (entry_price * position)
            else:
                pnl = 0
                pnl_pct = 0

            trades.append(
                {
                    "date": date,
                    "type": "sell",
                    "price": sell_price,
                    "shares": position,
                    "value": proceeds,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
            )

            position = 0

    # Close final position
    if position > 0 and len(data) > 0:
        final_price = data.iloc[-1]["close"]
        proceeds = position * final_price * (1 - commission)
        cash += proceeds
        equity_curve.append(cash)

    return trades, pd.Series(equity_curve)


def calculate_metrics(trades: list[dict], equity_curve: pd.Series) -> BacktestMetrics:
    """
    Calculate backtest metrics.

    Args:
        trades: List of executed trades
        equity_curve: Equity over time

    Returns:
        Backtest metrics
    """
    if len(equity_curve) == 0:
        return BacktestMetrics(
            total_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            avg_trade=0,
            best_trade=0,
            worst_trade=0,
            recovery_factor=0,
            calmar_ratio=0,
        )

    initial_capital = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital

    # Total return
    total_return = (final_capital - initial_capital) / initial_capital

    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Sharpe ratio
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    else:
        sharpe_ratio = 0

    # Max drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Trade statistics
    completed_trades = [t for t in trades if t.get("type") == "sell"]

    if completed_trades:
        wins = [t for t in completed_trades if t.get("pnl", 0) > 0]
        losses = [t for t in completed_trades if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(completed_trades)

        # Profit factor
        total_wins = sum(t.get("pnl", 0) for t in wins)
        total_losses = abs(sum(t.get("pnl", 0) for t in losses))
        profit_factor = (
            total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0
        )

        # Trade averages
        all_pnls = [t.get("pnl_pct", 0) for t in completed_trades]
        avg_trade = np.mean(all_pnls) if all_pnls else 0
        best_trade = max(all_pnls) if all_pnls else 0
        worst_trade = min(all_pnls) if all_pnls else 0
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade = 0
        best_trade = 0
        worst_trade = 0

    # Recovery factor
    recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

    # Calmar ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

    return BacktestMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
        avg_trade=avg_trade,
        best_trade=best_trade,
        worst_trade=worst_trade,
        recovery_factor=recovery_factor,
        calmar_ratio=calmar_ratio,
    )
