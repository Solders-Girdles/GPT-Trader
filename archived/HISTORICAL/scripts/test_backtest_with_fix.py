#!/usr/bin/env python3
"""Test backtest with the allocator fix."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.integration.orchestrator import IntegratedOrchestrator

def test_backtest():
    """Test if backtest now executes trades."""
    
    print("🚀 TESTING BACKTEST WITH ALLOCATOR FIX")
    print("="*50)
    
    # Create strategy
    strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,  # Relaxed parameters  
        atr_threshold_multiplier=0.8
    ))
    
    # Run backtest
    orchestrator = IntegratedOrchestrator()
    
    print("Running backtest...")
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n📊 Results:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Total return: {results.total_return*100:.2f}%")
    print(f"  Final value: ${100000 * (1 + results.total_return):,.2f}")
    
    if results.trades is not None and not results.trades.empty:
        print(f"\n💰 Trades Executed:")
        for i, trade in results.trades.iterrows():
            print(f"  {i+1}. {trade['symbol']}: "
                  f"Entry ${trade['entry_price']:.2f} → "
                  f"Exit ${trade.get('exit_price', 'Open'):.2f}, "
                  f"Qty: {trade['qty']}, "
                  f"PnL: ${trade.get('pnl', 0):.2f}")
    else:
        print(f"\n❌ No trades executed!")
        print(f"  This suggests the issue is elsewhere in the pipeline")

if __name__ == "__main__":
    test_backtest()