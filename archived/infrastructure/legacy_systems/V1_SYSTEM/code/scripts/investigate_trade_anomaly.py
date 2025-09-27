#!/usr/bin/env python3
"""
Investigate the anomaly: 0 trades reported but 75% returns.
This is impossible and indicates a serious calculation bug.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import pandas as pd
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def investigate_trade_anomaly():
    """Deep dive into the trade execution and return calculation."""
    
    print("="*80)
    print("INVESTIGATING TRADE ANOMALY: 0 Trades but 75% Returns")
    print("="*80)
    
    # Simple test configuration
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=True,
        apply_signal_filters=False,  # Disable to simplify
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    # Track key metrics before backtest
    print(f"\nBefore Backtest:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Current Equity: ${optimizer.current_equity:,.2f}")
    print(f"  Trades: {len(optimizer.trades) if hasattr(optimizer, 'trades') else 0}")
    
    # Run backtest
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    # Analyze results
    print(f"\nAfter Backtest:")
    print(f"  Final Equity: ${optimizer.current_equity:,.2f}")
    print(f"  Equity Change: ${optimizer.current_equity - config.initial_capital:,.2f}")
    print(f"  Trades Executed: {len(optimizer.trades) if hasattr(optimizer, 'trades') else 'NO TRADES ATTRIBUTE'}")
    
    # Check BacktestResults object
    print(f"\nResult Metrics:")
    print(f"  Total Return: {result['metrics']['total_return']:.2f}%")
    print(f"  Total Trades: {result['metrics']['total_trades']}")
    print(f"  Final Equity: ${result['metrics']['final_equity']:,.2f}")
    
    # Check equity history
    if hasattr(optimizer, 'equity_history'):
        print(f"\nEquity History:")
        print(f"  Entries: {len(optimizer.equity_history)}")
        if optimizer.equity_history:
            print(f"  First: ${optimizer.equity_history[0][1]:,.2f}")
            print(f"  Last: ${optimizer.equity_history[-1][1]:,.2f}")
            
            # Check for jumps in equity
            for i in range(1, min(10, len(optimizer.equity_history))):
                prev_equity = optimizer.equity_history[i-1][1]
                curr_equity = optimizer.equity_history[i][1]
                change = curr_equity - prev_equity
                if abs(change) > 0.01:  # Any change
                    print(f"  Day {i}: ${prev_equity:,.2f} -> ${curr_equity:,.2f} (${change:+,.2f})")
    
    # Check positions
    print(f"\nPositions:")
    print(f"  Current Positions: {optimizer.current_positions}")
    
    # Check trades in detail
    if hasattr(optimizer, 'trades') and optimizer.trades:
        print(f"\nTrade Details:")
        for i, trade in enumerate(optimizer.trades[:5]):
            print(f"  Trade {i+1}: {trade}")
    else:
        print("\nNO TRADES FOUND IN OPTIMIZER")
    
    # Check if trades are in the result
    if 'trades' in result:
        print(f"\nTrades in Result: {len(result['trades'])}")
        if result['trades']:
            print("First few trades:", result['trades'][:3])
    
    # Check the orchestrator parent class
    print(f"\nOrchestrator State:")
    print(f"  Has trades list: {hasattr(optimizer, 'trades')}")
    print(f"  Has ledger: {hasattr(optimizer, 'ledger')}")
    print(f"  Has portfolio: {hasattr(optimizer, 'portfolio')}")
    
    # Check errors
    if 'errors' in result and result['errors']:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors'][:5]:
            print(f"  - {error}")
    
    return result


def check_calculation_logic():
    """Check how returns are calculated."""
    
    print("\n" + "="*80)
    print("CHECKING RETURN CALCULATION LOGIC")
    print("="*80)
    
    # Simulate what might be happening
    initial = 10000
    final = 17576.06  # What we saw in the results
    
    calc_return = ((final - initial) / initial) * 100
    print(f"If initial=${initial} and final=${final}:")
    print(f"  Calculated return: {calc_return:.2f}%")
    
    # Check if equity is being updated without trades
    print("\nPossible causes of phantom returns:")
    print("1. Equity being updated by overnight P&L without actual trades")
    print("2. Position values being marked to market without trade records")
    print("3. _calculate_metrics using wrong equity values")
    print("4. Double counting or calculation error in equity updates")


if __name__ == "__main__":
    result = investigate_trade_anomaly()
    check_calculation_logic()
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    if result['metrics']['total_trades'] == 0 and result['metrics']['total_return'] > 0:
        print("❌ CRITICAL BUG: Returns without trades indicates equity calculation error")
        print("   Likely cause: Positions being valued without trade execution tracking")
    else:
        print("✓ Trade count matches execution")