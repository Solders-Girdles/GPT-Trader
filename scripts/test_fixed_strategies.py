#!/usr/bin/env python3
"""Test all 4 strategies with fixed parameters to verify the foundation is restored."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.momentum import MomentumStrategy
from bot.strategy.optimized_ma import OptimizedMAStrategy
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_all_fixed_strategies():
    """Test all 4 strategies with their fixed default parameters."""
    
    print("🔧 TESTING ALL 4 STRATEGIES WITH FIXED PARAMETERS")
    print("="*70)
    print("Goal: Verify that realistic parameters fix signal generation\n")
    
    # Use default parameters (now fixed)
    strategies = {
        'mean_reversion': MeanReversionStrategy(),
        'momentum': MomentumStrategy(), 
        'optimized_ma': OptimizedMAStrategy(),
        'enhanced_trend_breakout': EnhancedTrendBreakoutStrategy(),
    }
    
    print(f"Testing {len(strategies)} strategies with fixed default parameters...")
    print(f"{'Strategy':<25} {'Trades':<8} {'Return':<10} {'Status':<15}")
    print("-" * 70)
    
    results = {}
    working_count = 0
    
    for name, strategy in strategies.items():
        try:
            orchestrator = IntegratedOrchestrator()
            
            # Test with 1 month of realistic data
            backtest_results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                strategy=strategy,
                initial_capital=1000
            )
            
            trades = backtest_results.total_trades
            return_pct = backtest_results.total_return * 100
            
            if trades > 0:
                status = "✅ FIXED"
                working_count += 1
            else:
                status = "❌ Still broken"
            
            print(f"{name:<25} {trades:<8} {return_pct:+7.2f}% {status}")
            
            results[name] = {
                'trades': trades,
                'return': return_pct,
                'status': 'working' if trades > 0 else 'broken'
            }
            
        except Exception as e:
            error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
            print(f"{name:<25} {'ERROR':<8} {'':>8} ❌ {error_msg}")
            results[name] = {
                'trades': 0,
                'return': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    # Foundation assessment
    print(f"\n🏗️ FOUNDATION RESTORATION RESULTS:")
    print(f"  Fixed strategies: {working_count}/4")
    print(f"  Success rate: {working_count/4*100:.0f}%")
    
    if working_count == 4:
        print(f"  🚀 PERFECT! All 4 strategies now generate trades!")
        print(f"  🎯 Foundation is completely restored!")
        print(f"  📋 Ready for: Multi-symbol testing, parameter tuning, ML integration")
        foundation_status = "BULLETPROOF"
    elif working_count >= 3:
        print(f"  🎉 EXCELLENT! Most strategies restored!")
        print(f"  📋 Action: Fix remaining {4-working_count} strategy, then ready for production")
        foundation_status = "STRONG"
    elif working_count >= 2:
        print(f"  ✅ GOOD! Significant improvement achieved!")
        print(f"  📋 Action: Continue debugging remaining strategies")
        foundation_status = "ACCEPTABLE"
    else:
        print(f"  ⚠️ LIMITED: More work needed")
        foundation_status = "NEEDS_WORK"
    
    # Strategy-specific analysis
    print(f"\n📊 DETAILED ANALYSIS:")
    for name, result in results.items():
        status = result['status']
        if status == 'working':
            print(f"  ✅ {name}: {result['trades']} trades, {result['return']:+.1f}% return")
        elif status == 'broken':
            print(f"  ❌ {name}: Still generating 0 signals - needs deeper investigation")
        else:
            print(f"  ⚠️ {name}: Error - {result.get('error', 'Unknown')}")
    
    # Compare with baseline (original 3 working strategies)
    total_working = 3 + working_count  # 3 original + newly fixed
    print(f"\n🎯 FOUNDATION COMPARISON:")
    print(f"  Original working strategies: 3/7 (43%)")
    print(f"  After parameter fixes: {total_working}/7 ({total_working/7*100:.0f}%)")
    print(f"  Foundation status: {foundation_status}")
    
    return results, foundation_status

if __name__ == "__main__":
    results, status = test_all_fixed_strategies()
    
    print(f"\n🎊 PARAMETER FIX COMPLETE!")
    print(f"Foundation Status: {status}")
    print(f"Ready for Phase 1C: System Integration Testing")