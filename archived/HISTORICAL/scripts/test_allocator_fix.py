#!/usr/bin/env python3
"""Test the allocator fix directly."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dataflow.pipeline import DataPipeline
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.portfolio.allocator import PortfolioRules, allocate_signals

def test_allocator_fix():
    """Test if the allocator fix works."""
    
    print("🔧 TESTING ALLOCATOR FIX")
    print("="*50)
    
    # Load data and generate signals
    pipeline = DataPipeline()
    df = pipeline.get_data(["AAPL"], datetime(2024, 1, 1), datetime(2024, 6, 30))["AAPL"]
    
    strategy = VolatilityStrategy(VolatilityParams())
    signals = strategy.generate_signals(df)
    
    # Combine with market data
    combined = df.join(signals, how="left")
    
    print(f"📊 Signal Analysis:")
    print(f"  Total bars: {len(combined)}")
    print(f"  Last signal: {combined['signal'].iloc[-1]}")
    print(f"  Signals > 0: {(combined['signal'] > 0).sum()}")
    
    # Show last 25 signals
    print(f"\n📈 Last 25 Signal Values:")
    last_25_signals = combined['signal'].iloc[-25:]
    for i, (date, signal) in enumerate(last_25_signals.items()):
        if signal > 0:
            print(f"  {date}: {signal:.1f} ⭐")
        else:
            print(f"  {date}: {signal:.1f}")
    
    # Test allocator
    print(f"\n💼 Testing Allocator:")
    
    rules = PortfolioRules()
    signals_dict = {"AAPL": combined}
    
    print(f"  Portfolio rules:")
    print(f"    Max positions: {rules.max_positions}")
    print(f"    Risk per trade: {rules.per_trade_risk_pct*100:.1f}%")
    
    # Call allocator
    allocations = allocate_signals(signals_dict, 100000, rules)
    
    print(f"\n📊 Allocator Result:")
    if allocations:
        for symbol, qty in allocations.items():
            position_value = qty * combined['close'].iloc[-1]
            print(f"  ✅ {symbol}: {qty} shares (${position_value:,.2f})")
            print(f"     Entry reason: Found signal in recent window")
    else:
        print(f"  ❌ No allocations - Fix didn't work!")
        
        # Debug why
        print(f"\n🔍 Debug Info:")
        recent_signals = combined['signal'].iloc[-120:]  # Match allocator window
        active_recent = recent_signals[recent_signals > 0]
        
        print(f"  Recent signals (last 120): {recent_signals.sum():.1f}")
        print(f"  Active recent signals: {len(active_recent)}")
        
        if len(active_recent) > 0:
            print(f"  Most recent active: {active_recent.iloc[-1]:.1f} on {active_recent.index[-1]}")
        else:
            print(f"  No active signals in last 120 bars")
            
            # Look further back
            all_signals = combined['signal'][combined['signal'] > 0]
            if len(all_signals) > 0:
                last_signal_date = all_signals.index[-1]
                bars_ago = len(combined) - combined.index.get_loc(last_signal_date) - 1
                print(f"  Last signal was {bars_ago} bars ago on {last_signal_date}")

if __name__ == "__main__":
    test_allocator_fix()