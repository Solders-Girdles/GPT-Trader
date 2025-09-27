#!/usr/bin/env python3
"""Simple test of relaxed parameter strategies - focused results."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.mean_reversion import MeanReversionStrategy, MeanReversionParams
from bot.strategy.momentum import MomentumStrategy, MomentumParams
from bot.strategy.optimized_ma import OptimizedMAStrategy, OptimizedMAParams
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_one_strategy(name, strategy):
    """Test a single strategy and return clean results."""
    
    try:
        orchestrator = IntegratedOrchestrator()
        
        results = orchestrator.run_backtest(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy=strategy,
            initial_capital=1000
        )
        
        return {
            'trades': results.total_trades,
            'return': results.total_return * 100,
            'status': 'working' if results.total_trades > 0 else 'no_trades'
        }
        
    except Exception as e:
        return {
            'trades': 0,
            'return': 0.0,
            'status': 'error',
            'error': str(e)[:50]
        }

def main():
    print("🔧 SIMPLE TEST: RELAXED PARAMETER STRATEGIES")
    print("="*60)
    
    # Create relaxed strategies
    strategies = {
        'mean_reversion_relaxed': MeanReversionStrategy(MeanReversionParams(
            rsi_period=10, oversold_threshold=40.0, overbought_threshold=60.0
        )),
        'momentum_relaxed': MomentumStrategy(MomentumParams(
            roc_period=10, momentum_threshold=0.015, volume_threshold=1.2
        )),
        'optimized_ma_relaxed': OptimizedMAStrategy(OptimizedMAParams(
            fast=5, slow=15, volume_filter=False, rsi_filter=False, trend_strength_filter=False
        )),
        'enhanced_trend_default': EnhancedTrendBreakoutStrategy(),
    }
    
    print(f"Testing {len(strategies)} strategies...")
    print(f"{'Strategy':<25} {'Trades':<8} {'Return':<10} {'Status':<15}")
    print("-" * 70)
    
    results = {}
    working_count = 0
    
    for name, strategy in strategies.items():
        result = test_one_strategy(name, strategy)
        results[name] = result
        
        trades = result['trades']
        return_pct = result['return']
        status_text = result['status']
        
        if status_text == 'working':
            status_display = "✅ WORKING"
            working_count += 1
        elif status_text == 'no_trades':
            status_display = "❌ No trades"
        else:
            status_display = f"❌ {result.get('error', 'Error')[:10]}"
        
        print(f"{name:<25} {trades:<8} {return_pct:+7.2f}% {status_display}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Working strategies: {working_count}/4")
    print(f"  Success rate: {working_count/4*100:.0f}%")
    
    if working_count >= 3:
        print(f"  🎉 EXCELLENT! Relaxed parameters fix most strategies!")
    elif working_count >= 2:
        print(f"  ✅ GOOD! Significant improvement!")
    elif working_count >= 1:
        print(f"  ⚠️ PARTIAL: Some progress made")
    else:
        print(f"  ❌ POOR: Need different approach")
    
    return results

if __name__ == "__main__":
    main()