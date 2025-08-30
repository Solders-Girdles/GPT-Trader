#!/usr/bin/env python3
"""Test the fixed backtest system."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.integration.orchestrator import IntegratedOrchestrator

def test_fixed_backtest():
    """Test backtest with position closing fix."""
    
    print("🚀 TESTING FIXED BACKTEST SYSTEM")
    print("="*50)
    
    orchestrator = IntegratedOrchestrator()
    
    # Test 1: Trend Breakout Strategy
    print("\n📈 Test 1: Trend Breakout Strategy")
    print("-" * 30)
    
    strategy1 = TrendBreakoutStrategy()
    results1 = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        strategy=strategy1,
        initial_capital=100000
    )
    
    print(f"Results: {results1.total_trades} trades, {results1.total_return*100:.2f}% return")
    
    # Test 2: Volatility Strategy
    print("\n🌊 Test 2: Volatility Strategy")
    print("-" * 30)
    
    strategy2 = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,
        atr_threshold_multiplier=0.8
    ))
    results2 = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        strategy=strategy2,
        initial_capital=100000
    )
    
    print(f"Results: {results2.total_trades} trades, {results2.total_return*100:.2f}% return")
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("="*30)
    print(f"Trend Breakout: {results1.total_trades} trades")
    print(f"Volatility:     {results2.total_trades} trades")
    
    if results1.total_trades > 0 or results2.total_trades > 0:
        print(f"\n✅ SUCCESS: Trades are now being recorded!")
        print(f"   The position closing fix is working correctly.")
        
        if results1.trades is not None and not results1.trades.empty:
            print(f"\n🔍 Trend Breakout Sample Trade:")
            first_trade = results1.trades.iloc[0]
            print(f"   Entry: ${first_trade['entry_price']:.2f} → Exit: ${first_trade['exit_price']:.2f}")
            print(f"   PnL: ${first_trade['pnl']:.2f}, Return: {first_trade['rtn']*100:.2f}%")
            
        if results2.trades is not None and not results2.trades.empty:
            print(f"\n🔍 Volatility Sample Trade:")
            first_trade = results2.trades.iloc[0]
            print(f"   Entry: ${first_trade['entry_price']:.2f} → Exit: ${first_trade['exit_price']:.2f}")
            print(f"   PnL: ${first_trade['pnl']:.2f}, Return: {first_trade['rtn']*100:.2f}%")
    else:
        print(f"\n❌ Still no trades recorded")

if __name__ == "__main__":
    test_fixed_backtest()