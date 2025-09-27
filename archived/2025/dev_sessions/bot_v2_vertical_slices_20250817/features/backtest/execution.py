"""
Trade execution simulation for backtesting.
"""

from typing import List, Dict, Tuple
import pandas as pd


def simulate_trades(
    signals: pd.Series,
    data: pd.DataFrame,
    initial_capital: float,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 0.95
) -> Tuple[List[Dict], pd.Series, pd.Series]:
    """
    Simulate trade execution based on signals.
    
    Args:
        signals: Trading signals
        data: Market data
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate
        position_size: Fraction of capital to use
        
    Returns:
        Tuple of (trades, equity_curve, returns)
    """
    trades = []
    cash = initial_capital
    position_qty = 0
    equity_curve = []
    daily_returns = []
    trade_id = 0
    
    for i, (date, signal) in enumerate(signals.items()):
        current_price = data.loc[date, 'close']
        
        # Calculate current equity
        current_equity = cash + (position_qty * current_price)
        equity_curve.append(current_equity)
        
        # Calculate daily return
        if i > 0:
            prev_equity = equity_curve[i-1]
            daily_return = (current_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
        else:
            daily_returns.append(0)
        
        # Process buy signal
        if signal == 1 and position_qty == 0:
            buy_price = current_price * (1 + slippage)
            max_qty = int((cash * position_size) / buy_price)
            
            if max_qty > 0:
                cost = max_qty * buy_price * (1 + commission)
                cash -= cost
                position_qty = max_qty
                
                trades.append({
                    'id': trade_id,
                    'date': date,
                    'side': 'buy',
                    'price': buy_price,
                    'quantity': max_qty,
                    'commission': cost * commission
                })
                trade_id += 1
        
        # Process sell signal
        elif signal == -1 and position_qty > 0:
            sell_price = current_price * (1 - slippage)
            proceeds = position_qty * sell_price * (1 - commission)
            cash += proceeds
            
            trades.append({
                'id': trade_id,
                'date': date,
                'side': 'sell',
                'price': sell_price,
                'quantity': position_qty,
                'commission': proceeds * commission
            })
            trade_id += 1
            position_qty = 0
    
    # Close any remaining position
    if position_qty > 0:
        final_price = data.iloc[-1]['close']
        proceeds = position_qty * final_price * (1 - commission)
        cash += proceeds
        
        trades.append({
            'id': trade_id,
            'date': data.index[-1],
            'side': 'sell',
            'price': final_price,
            'quantity': position_qty,
            'commission': proceeds * commission
        })
    
    equity_series = pd.Series(equity_curve, index=data.index)
    returns_series = pd.Series(daily_returns, index=data.index)
    
    return trades, equity_series, returns_series