"""
Main backtest orchestration - entry point for the slice.
"""

from datetime import datetime
from typing import Optional
from .types import BacktestResult
from .data import fetch_historical_data
from .signals import generate_signals
from .execution import simulate_trades
from .metrics import calculate_metrics


def run_backtest(
    strategy: str,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_capital: float = 10000,
    commission: float = 0.001,
    slippage: float = 0.0005,
    **strategy_params
) -> BacktestResult:
    """
    Run a complete backtest.
    
    This is the main entry point for the backtest feature slice.
    Everything needed for backtesting is contained within this slice.
    
    Args:
        strategy: Name of strategy to test
        symbol: Stock symbol
        start: Start date
        end: End date
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate
        **strategy_params: Additional strategy parameters
        
    Returns:
        BacktestResult with trades, metrics, and equity curve
    """
    # Step 1: Fetch historical data
    data = fetch_historical_data(symbol, start, end)
    
    # Step 2: Generate trading signals
    signals = generate_signals(strategy, data, **strategy_params)
    
    # Step 3: Simulate trade execution
    trades, equity_curve, returns = simulate_trades(
        signals, data, initial_capital, commission, slippage
    )
    
    # Step 4: Calculate performance metrics
    metrics = calculate_metrics(trades, equity_curve, returns, initial_capital)
    
    # Step 5: Return complete results
    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        returns=returns,
        metrics=metrics
    )