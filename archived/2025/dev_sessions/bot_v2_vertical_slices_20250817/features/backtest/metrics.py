"""
Performance metrics calculation for backtesting.
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from .types import BacktestMetrics


def calculate_metrics(
    trades: List[Dict],
    equity_curve: pd.Series,
    returns: pd.Series,
    initial_capital: float
) -> BacktestMetrics:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        trades: List of executed trades
        equity_curve: Portfolio equity over time
        returns: Daily returns
        initial_capital: Starting capital
        
    Returns:
        BacktestMetrics object
    """
    metrics = BacktestMetrics()
    
    # Total return
    if len(equity_curve) > 1:
        final_value = equity_curve.iloc[-1]
        metrics.total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Sharpe ratio (simplified - 0 risk-free rate)
    if len(returns) > 1 and returns.std() > 0:
        metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # Maximum drawdown
    if len(equity_curve) > 1:
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        metrics.max_drawdown = abs(drawdown.min())
    
    # Trade statistics
    if trades:
        metrics.total_trades = len([t for t in trades if t['side'] == 'buy'])
        
        # Calculate P&L for each round trip
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell']
        
        if buy_trades and sell_trades:
            wins = 0
            total_profit = 0
            total_loss = 0
            
            for i, buy in enumerate(buy_trades):
                if i < len(sell_trades):
                    sell = sell_trades[i]
                    pnl = (sell['price'] - buy['price']) * buy['quantity']
                    
                    if pnl > 0:
                        wins += 1
                        total_profit += pnl
                    else:
                        total_loss += abs(pnl)
            
            if metrics.total_trades > 0:
                metrics.win_rate = (wins / metrics.total_trades) * 100
            
            if total_loss > 0:
                metrics.profit_factor = total_profit / total_loss
    
    return metrics