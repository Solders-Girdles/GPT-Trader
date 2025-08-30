#!/usr/bin/env python3
"""Debug what columns are passed to the allocator for different strategies."""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dataflow.pipeline import DataPipeline
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge

def debug_strategy_columns(strategy_name: str, strategy_instance):
    """Debug what columns a strategy produces and what the allocator receives."""
    
    print(f"\n{'='*60}")
    print(f"🔍 {strategy_name}")
    print('='*60)
    
    # Load data
    pipeline = DataPipeline()
    df = pipeline.get_data(["AAPL"], datetime(2024, 1, 1), datetime(2024, 6, 30))["AAPL"]
    
    # Generate signals
    signals = strategy_instance.generate_signals(df)
    
    # Show columns from strategy
    print(f"\n📊 Strategy Output Columns:")
    for col in signals.columns:
        # Show column name and some sample values
        non_zero = signals[col][signals[col] != 0]
        if len(non_zero) > 0:
            sample = non_zero.iloc[-1] if not non_zero.empty else 0
            if isinstance(sample, (int, float)):
                print(f"  - {col}: {type(sample).__name__} (last non-zero: {sample:.4f})")
            else:
                print(f"  - {col}: {type(sample).__name__} (last non-zero: {sample})")
        else:
            print(f"  - {col}: all zeros")
    
    # Create bridge and process
    rules = PortfolioRules()
    bridge = StrategyAllocatorBridge(strategy_instance, rules)
    
    # Process signals through bridge
    market_data = {"AAPL": df}
    allocations = bridge.process_signals(market_data, 100000)
    
    # Show what the allocator sees
    # Let's manually recreate what the bridge does
    combined = df.join(signals, how="left")
    
    print(f"\n📦 Combined Data for Allocator:")
    print(f"  Total columns: {len(combined.columns)}")
    print(f"  Columns: {list(combined.columns)}")
    
    # Check key columns the allocator needs
    print(f"\n✅ Required Columns Check:")
    required = ["close", "signal", "atr"]
    for col in required:
        if col in combined.columns:
            last_val = combined[col].iloc[-1]
            if pd.notna(last_val):
                print(f"  ✅ {col}: {last_val:.4f}")
            else:
                print(f"  ✅ {col}: NaN")
        else:
            print(f"  ❌ {col}: MISSING")
    
    # Check optional columns
    print(f"\n📋 Optional Columns Check:")
    optional = ["donchian_upper", "open"]
    for col in optional:
        if col in combined.columns:
            last_val = combined[col].iloc[-1]
            if pd.notna(last_val):
                print(f"  ✅ {col}: {last_val:.4f}")
            else:
                print(f"  ✅ {col}: NaN")
        else:
            print(f"  ➖ {col}: not present")
    
    # Show allocations result
    print(f"\n💰 Allocation Result:")
    if allocations:
        for sym, qty in allocations.items():
            if qty > 0:
                print(f"  {sym}: {qty} shares")
    else:
        print(f"  ⚠️  No allocations!")
        
        # Debug why no allocation
        print(f"\n🔍 Debugging No Allocation:")
        
        # Check signal
        last_signal = combined["signal"].iloc[-1] if "signal" in combined.columns else None
        print(f"  Last signal: {last_signal}")
        
        # Check ATR
        last_atr = combined["atr"].iloc[-1] if "atr" in combined.columns else None
        print(f"  Last ATR: {last_atr}")
        
        # Check price
        last_price = combined["close"].iloc[-1] if "close" in combined.columns else None
        print(f"  Last price: {last_price}")
        
        # Calculate position size manually
        if last_signal and last_signal > 0 and last_atr and last_atr > 0:
            from bot.portfolio.allocator import position_size
            qty = position_size(100000, last_atr, last_price, rules)
            print(f"  Manual position size calculation: {qty} shares")
            if qty == 0:
                print(f"    Risk USD: {100000 * rules.per_trade_risk_pct:.2f}")
                print(f"    Stop distance: {rules.atr_k * last_atr:.2f}")
                print(f"    Qty = Risk / Stop = {100000 * rules.per_trade_risk_pct:.2f} / {rules.atr_k * last_atr:.2f}")

def main():
    """Debug columns for all strategies."""
    
    print("📊 ALLOCATOR COLUMN DEBUGGER")
    print("="*60)
    
    # Test volatility with default parameters
    vol_default = VolatilityStrategy(VolatilityParams())
    debug_strategy_columns("Volatility (Default)", vol_default)
    
    # Test volatility with relaxed parameters  
    vol_relaxed = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,
        atr_threshold_multiplier=0.8
    ))
    debug_strategy_columns("Volatility (Relaxed)", vol_relaxed)
    
    # Test Demo MA
    ma_strategy = DemoMAStrategy(fast=10, slow=20)
    debug_strategy_columns("Demo MA", ma_strategy)
    
    # Test Trend Breakout
    trend_strategy = TrendBreakoutStrategy()
    debug_strategy_columns("Trend Breakout", trend_strategy)

if __name__ == "__main__":
    main()