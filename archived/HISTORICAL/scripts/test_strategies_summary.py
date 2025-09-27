#!/usr/bin/env python3
"""Quick summary test of all strategies with dynamic risk management."""

import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.demo_ma import DemoMAStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def quick_strategy_test():
    """Quick test of each strategy with $1K portfolio."""
    
    print("🎯 QUICK STRATEGY VALIDATION (All with $1K portfolio)")
    print("="*60)
    
    strategies_to_test = [
        ("trend_breakout", TrendBreakoutStrategy()),
        ("volatility", VolatilityStrategy(VolatilityParams(bb_std_dev=1.5, atr_threshold_multiplier=0.8))),
        ("demo_ma", DemoMAStrategy()),
    ]
    
    print(f"{'Strategy':<20} {'Trades':<8} {'Return':<10} {'Status':<15}")
    print("-" * 60)
    
    for strategy_name, strategy in strategies_to_test:
        try:
            orchestrator = IntegratedOrchestrator()
            
            results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),  # Just January 2024
                strategy=strategy,
                initial_capital=1000
            )
            
            trades = results.total_trades
            returns = results.total_return * 100
            
            if trades > 0:
                status = "✅ Working"
            else:
                status = "⚠️ No trades"
            
            print(f"{strategy_name:<20} {trades:<8} {returns:>+6.1f}%   {status:<15}")
            
        except Exception as e:
            error_short = str(e)[:20] + "..." if len(str(e)) > 20 else str(e)
            print(f"{strategy_name:<20} {'ERROR':<8} {'N/A':<10} ❌ {error_short}")
    
    print(f"\n🎯 DYNAMIC RISK EVIDENCE:")
    print("Check logs above for '[Dynamic limit for $1,000 portfolio]' messages")

if __name__ == "__main__":
    quick_strategy_test()