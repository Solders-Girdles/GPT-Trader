#!/usr/bin/env python3
"""Debug why orchestrator might be using wrong dates."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.dataflow.pipeline import DataPipeline
from bot.integration.orchestrator import IntegratedOrchestrator

def debug_date_loading():
    """Debug what dates are being loaded and processed."""
    
    print("📅 DATE LOADING DEBUG")
    print("="*40)
    
    # Test 1: Direct data pipeline
    print("\n📊 Test 1: DataPipeline direct")
    pipeline = DataPipeline()
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    data = pipeline.get_data(["AAPL"], start_date, end_date)
    
    if "AAPL" in data:
        df = data["AAPL"]
        print(f"  Data shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  First 3 dates: {df.index[:3].tolist()}")
        print(f"  Last 3 dates: {df.index[-3:].tolist()}")
    else:
        print("  ❌ No AAPL data loaded")
    
    # Test 2: Orchestrator data loading
    print("\n🎬 Test 2: Orchestrator data loading")
    orchestrator = IntegratedOrchestrator()
    
    # Create a minimal strategy
    strategy = VolatilityStrategy(VolatilityParams())
    
    # Patch the _load_market_data method to see what it does
    original_load = orchestrator._load_market_data
    
    def debug_load_market_data(symbols, results):
        print(f"  _load_market_data called with:")
        print(f"    symbols: {symbols}")
        print(f"    config start_date: {orchestrator.config.start_date}")
        print(f"    config end_date: {orchestrator.config.end_date}")
        
        market_data = original_load(symbols, results)
        
        print(f"  Market data loaded:")
        for symbol, df in market_data.items():
            print(f"    {symbol}: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        
        return market_data
    
    orchestrator._load_market_data = debug_load_market_data
    
    # Run backtest
    print(f"\n  Running backtest with dates {start_date} to {end_date}")
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n  Backtest results:")
    print(f"    Total trades: {results.total_trades}")
    print(f"    Start date: {results.start_date}")
    print(f"    End date: {results.end_date}")
    print(f"    Duration: {results.duration_days} days")

if __name__ == "__main__":
    debug_date_loading()