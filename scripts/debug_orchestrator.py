#!/usr/bin/env python3
"""Debug what the orchestrator is doing."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.dataflow.pipeline import DataPipeline
from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.portfolio.allocator import PortfolioRules

def debug_orchestrator_logic():
    """Debug what the orchestrator does step by step."""
    
    print("🔍 DEBUGGING ORCHESTRATOR LOGIC")
    print("="*50)
    
    # Step 1: Load data (like orchestrator does)
    pipeline = DataPipeline()
    market_data = pipeline.get_data(["AAPL"], datetime(2024, 1, 1), datetime(2024, 6, 30))
    
    print(f"📊 Market data loaded:")
    print(f"  AAPL: {len(market_data['AAPL'])} bars")
    
    # Step 2: Create strategy and bridge (like orchestrator does)
    strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,  # Relaxed parameters
        atr_threshold_multiplier=0.8
    ))
    
    rules = PortfolioRules()
    bridge = StrategyAllocatorBridge(strategy, rules)
    
    print(f"\n🔧 Strategy and bridge created")
    
    # Step 3: Simulate orchestrator daily loop for a few key dates
    df = market_data["AAPL"]
    
    # Find dates where signals were generated
    signals_full = strategy.generate_signals(df)
    signal_dates = signals_full[signals_full["signal"] > 0].index
    
    print(f"\n📈 Signal dates found: {len(signal_dates)}")
    for i, date in enumerate(signal_dates[:3]):
        print(f"  {i+1}. {date}")
    
    # Test allocation on the last trading day (like the real backtest does)
    last_date = df.index[-1]
    print(f"\n📅 Testing allocation on last trading day: {last_date}")
    
    # Get data up to that date (like _get_daily_data does)
    mask = df.index <= last_date
    daily_data = {"AAPL": df[mask].copy()}
    
    print(f"  Historical data provided: {len(daily_data['AAPL'])} bars")
    print(f"  Date range: {daily_data['AAPL'].index[0]} to {daily_data['AAPL'].index[-1]}")
    
    # Process signals through bridge (like orchestrator does)
    equity = 100000
    allocations = bridge.process_signals(daily_data, equity)
    
    print(f"\n💼 Bridge allocations:")
    if allocations:
        for symbol, qty in allocations.items():
            if qty > 0:
                price = daily_data[symbol]["close"].iloc[-1]
                value = qty * price
                print(f"  ✅ {symbol}: {qty} shares (${value:,.2f})")
            else:
                print(f"  ❌ {symbol}: No allocation")
    else:
        print(f"  ❌ No allocations returned")
    
    # Test on a signal date too
    if len(signal_dates) > 0:
        signal_date = signal_dates[0]
        print(f"\n📅 Testing allocation on first signal date: {signal_date}")
        
        # Get data up to signal date
        mask = df.index <= signal_date
        signal_day_data = {"AAPL": df[mask].copy()}
        
        print(f"  Historical data provided: {len(signal_day_data['AAPL'])} bars")
        
        allocations_signal_day = bridge.process_signals(signal_day_data, equity)
        
        print(f"\n💼 Bridge allocations on signal day:")
        if allocations_signal_day:
            for symbol, qty in allocations_signal_day.items():
                if qty > 0:
                    price = signal_day_data[symbol]["close"].iloc[-1]
                    value = qty * price
                    print(f"  ✅ {symbol}: {qty} shares (${value:,.2f})")
                else:
                    print(f"  ❌ {symbol}: No allocation")
        else:
            print(f"  ❌ No allocations returned")

if __name__ == "__main__":
    debug_orchestrator_logic()