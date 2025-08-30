#!/usr/bin/env python3
"""Debug the full orchestrator execution to find where trades disappear."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.integration.orchestrator import IntegratedOrchestrator

def debug_full_orchestrator():
    """Debug the full orchestrator execution with detailed logging."""
    
    print("🚀 DEBUGGING FULL ORCHESTRATOR EXECUTION")
    print("="*60)
    
    # Create strategy
    strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,  # Relaxed parameters  
        atr_threshold_multiplier=0.8
    ))
    
    print(f"📊 Strategy: {strategy.name}")
    print(f"  Parameters: BB std_dev={strategy.params.bb_std_dev}, ATR threshold={strategy.params.atr_threshold_multiplier}")
    
    # Create orchestrator and monkey-patch some methods to add logging
    orchestrator = IntegratedOrchestrator()
    
    # Store original methods
    original_execute_trades = orchestrator._execute_trades
    original_validate_allocations = orchestrator.risk_integration.validate_allocations
    
    # Create logging wrappers
    def logged_execute_trades(target_allocations, trade_date):
        print(f"\n🔄 _execute_trades called for {trade_date}:")
        print(f"  Target allocations: {target_allocations}")
        
        result = original_execute_trades(target_allocations, trade_date)
        
        print(f"  Trades executed: {result}")
        print(f"  Current positions after: {orchestrator.current_positions}")
        
        return result
    
    def logged_validate_allocations(**kwargs):
        allocations = kwargs.get('allocations', {})
        print(f"\n🛡️ Risk validation called:")
        print(f"  Input allocations: {allocations}")
        
        result = original_validate_allocations(**kwargs)
        
        print(f"  Risk-adjusted allocations: {result.adjusted_allocations}")
        print(f"  Risk warnings: {result.warnings}")
        print(f"  Risk errors: {result.errors}")
        
        return result
    
    # Apply monkey patches
    orchestrator._execute_trades = logged_execute_trades
    orchestrator.risk_integration.validate_allocations = logged_validate_allocations
    
    print(f"\n🏃 Running backtest...")
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 2, 14),  # Start from first signal date
        end_date=datetime(2024, 2, 16),    # Just a few days for detailed debugging
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n📊 Final Results:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Total return: {results.total_return*100:.2f}%")
    print(f"  Final value: ${100000 * (1 + results.total_return):,.2f}")
    
    if results.trades is not None and not results.trades.empty:
        print(f"\n💰 Trades in results:")
        for i, trade in results.trades.iterrows():
            print(f"  {i+1}. {trade}")
    else:
        print(f"\n❌ No trades in results dataframe")
    
    print(f"\n📈 Equity history:")
    for date, equity in results.equity_curve.items() if results.equity_curve else []:
        print(f"  {date}: ${equity:,.2f}")

if __name__ == "__main__":
    debug_full_orchestrator()