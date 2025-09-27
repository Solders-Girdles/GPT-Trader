#!/usr/bin/env python3
"""Test backtest with short date range that should have signals."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.integration.orchestrator import IntegratedOrchestrator

def test_short_range():
    """Test backtest with date range that should have signals."""
    
    print("🎯 TESTING SHORT DATE RANGE WITH SIGNALS")
    print("="*50)
    
    # Create strategy
    strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,  # Relaxed parameters  
        atr_threshold_multiplier=0.8
    ))
    
    # Run backtest for period where we know signals occur
    orchestrator = IntegratedOrchestrator()
    
    # Use the same date range as the debug script that worked
    start_date = datetime(2024, 2, 14)
    end_date = datetime(2024, 2, 16)
    
    print(f"Running backtest for {start_date.date()} to {end_date.date()}...")
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n📊 Results:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Total return: {results.total_return*100:.2f}%")
    print(f"  Duration: {results.duration_days} days")
    print(f"  Start: {results.start_date}")
    print(f"  End: {results.end_date}")
    
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
    
    # Test longer range that includes signal period
    print(f"\n" + "="*50)
    print("🚀 TESTING EXTENDED RANGE (Feb-Mar 2024)")
    print("="*50)
    
    extended_results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 3, 31),
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n📊 Extended Results:")
    print(f"  Total trades: {extended_results.total_trades}")
    print(f"  Total return: {extended_results.total_return*100:.2f}%")
    print(f"  Duration: {extended_results.duration_days} days")

if __name__ == "__main__":
    test_short_range()