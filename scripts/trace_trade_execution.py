#!/usr/bin/env python3
"""
Trace exactly what happens during trade execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def trace_execution():
    """Trace the execution path to see where trades are lost."""
    
    print("="*80)
    print("TRACING TRADE EXECUTION")
    print("="*80)
    
    # Minimal configuration
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),  # Just 5 days
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=False,
        apply_signal_filters=False,
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False  # Use parent's _execute_trades
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    # Monkey-patch to trace execution
    original_execute = optimizer._execute_trades
    calls_to_execute = []
    
    def traced_execute(allocations, date):
        print(f"\n_execute_trades called with:")
        print(f"  Allocations: {allocations}")
        print(f"  Date: {date}")
        print(f"  Current positions: {optimizer.current_positions}")
        
        # Call original
        result = original_execute(allocations, date)
        
        print(f"  Result: {result} trades executed")
        print(f"  New positions: {optimizer.current_positions}")
        print(f"  Ledger size: {len(optimizer.ledger.to_trades_dataframe())}")
        
        calls_to_execute.append({
            'allocations': allocations,
            'date': date,
            'result': result
        })
        
        return result
    
    optimizer._execute_trades = traced_execute
    
    # Run backtest
    print("\nRunning backtest...")
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total calls to _execute_trades: {len(calls_to_execute)}")
    print(f"Final positions: {optimizer.current_positions}")
    print(f"Final ledger entries: {len(optimizer.ledger.to_trades_dataframe())}")
    print(f"Result total_trades: {result['metrics']['total_trades']}")
    
    # Check ledger content
    trades_df = optimizer.ledger.to_trades_dataframe()
    if not trades_df.empty:
        print("\nLedger content:")
        print(trades_df)
    else:
        print("\nLedger is empty!")
    
    # Check if ledger has fills
    print(f"\nLedger fills: {len(optimizer.ledger.fills)}")
    print(f"Ledger orders: {len(optimizer.ledger.orders)}")
    print(f"Ledger trades: {len(optimizer.ledger.trades)}")
    
    return result


if __name__ == "__main__":
    trace_execution()