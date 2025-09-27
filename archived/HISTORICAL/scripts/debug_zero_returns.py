#!/usr/bin/env python3
"""
Debug why strategies are showing 0% returns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import logging

from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
from bot.strategy.mean_reversion import MeanReversionStrategy, MeanReversionParams

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_backtest():
    """Run a simple backtest and debug the return calculation."""
    
    print("="*60)
    print("DEBUGGING ZERO RETURNS ISSUE")
    print("="*60)
    
    # Simple configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        initial_capital=10000,
        quiet_mode=False
    )
    
    # Create orchestrator and strategy
    orchestrator = IntegratedOrchestrator(config)
    strategy = MeanReversionStrategy(MeanReversionParams(
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70
    ))
    
    print(f"\nInitial Capital: ${config.initial_capital:,.2f}")
    print(f"Initial Equity: ${orchestrator.current_equity:,.2f}")
    
    # Run backtest
    result = orchestrator.run_backtest(strategy, ["AAPL"])
    
    # Debug the results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Check trades
    print(f"\nNumber of trades: {len(orchestrator.trades)}")
    if orchestrator.trades:
        print("\nFirst 5 trades:")
        for i, trade in enumerate(orchestrator.trades[:5]):
            print(f"  Trade {i+1}: {trade['date']} {trade['action']} {trade['shares']} "
                  f"{trade['symbol']} @ ${trade['price']:.2f}")
    
    # Check final state
    print(f"\nFinal Equity: ${orchestrator.current_equity:,.2f}")
    print(f"P&L: ${orchestrator.current_equity - config.initial_capital:,.2f}")
    
    # Check metrics
    metrics = result.get('metrics', {})
    print(f"\nMetrics from result:")
    print(f"  Total Return: {metrics.get('total_return', 'N/A')}%")
    print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
    print(f"  Final Equity: ${metrics.get('final_equity', 'N/A'):,.2f}")
    
    # Check if _calculate_metrics is being called
    print("\n" + "="*60)
    print("INVESTIGATING CALCULATION")
    print("="*60)
    
    # Manually calculate what the return should be
    if config.initial_capital > 0:
        manual_return = ((orchestrator.current_equity - config.initial_capital) / config.initial_capital) * 100
        print(f"Manual calculation: {manual_return:.2f}%")
    
    # Check if trades are being executed
    if len(orchestrator.trades) == 0:
        print("\n⚠️ NO TRADES EXECUTED - This is why return is 0%")
        print("Possible reasons:")
        print("  1. Strategy not generating signals")
        print("  2. Allocator not converting signals to positions")
        print("  3. Risk management blocking all trades")
        print("  4. Insufficient capital for minimum position size")
    
    # Check positions
    print(f"\nCurrent Positions: {orchestrator.current_positions}")
    
    # Check if there's a calculation bug
    if hasattr(orchestrator, 'equity_curve'):
        print(f"\nEquity Curve Length: {len(orchestrator.equity_curve)}")
        if orchestrator.equity_curve:
            print(f"First value: ${orchestrator.equity_curve[0]:,.2f}")
            print(f"Last value: ${orchestrator.equity_curve[-1]:,.2f}")


if __name__ == "__main__":
    debug_backtest()