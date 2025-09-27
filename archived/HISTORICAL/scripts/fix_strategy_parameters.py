#!/usr/bin/env python3
"""Fix the 4 failing strategies with relaxed, more realistic parameters."""

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

def create_relaxed_strategies():
    """Create strategies with relaxed, more realistic parameters."""
    
    print("🎛️ CREATING RELAXED PARAMETER STRATEGIES")
    print("="*60)
    
    strategies = {}
    
    # 1. Mean Reversion - More sensitive RSI levels
    relaxed_mr_params = MeanReversionParams(
        rsi_period=10,           # Shorter period for faster signals
        oversold_threshold=40.0,  # Less extreme oversold (was 30)
        overbought_threshold=60.0, # Less extreme overbought (was 70)
        exit_rsi_threshold=50.0   # Exit at neutral RSI
    )
    strategies['mean_reversion_relaxed'] = MeanReversionStrategy(relaxed_mr_params)
    print(f"  ✅ Mean Reversion: RSI(10) 40/60 thresholds (was 30/70)")
    
    # 2. Momentum - Lower momentum and volume requirements  
    relaxed_momentum_params = MomentumParams(
        roc_period=10,            # Shorter lookback (was 20)
        momentum_threshold=0.015, # 1.5% momentum (was 3.0%)
        volume_threshold=1.2,     # 1.2x volume (was 1.5x)
        momentum_exit_threshold=0.005  # 0.5% exit (was 1.0%)
    )
    strategies['momentum_relaxed'] = MomentumStrategy(relaxed_momentum_params)
    print(f"  ✅ Momentum: 1.5% threshold, 1.2x volume (was 3.0%, 1.5x)")
    
    # 3. Optimized MA - Disable filters, use closer MAs
    relaxed_ma_params = OptimizedMAParams(
        fast=5,                   # Faster MA (was 10)
        slow=15,                  # Closer slow MA (was 20)
        volume_filter=False,      # Disable volume filter
        rsi_filter=False,         # Disable RSI filter
        trend_strength_filter=False,  # Disable trend filter
        volatility_adjustment=False   # Disable volatility adjustment
    )
    strategies['optimized_ma_relaxed'] = OptimizedMAStrategy(relaxed_ma_params)
    print(f"  ✅ Optimized MA: MA(5,15), all filters disabled (was MA(10,20) + filters)")
    
    # 4. Enhanced Trend Breakout - Use default but test separately
    # Note: May need to examine this strategy's parameters separately
    strategies['enhanced_trend_relaxed'] = EnhancedTrendBreakoutStrategy()
    print(f"  ⚠️ Enhanced Trend: Using defaults (needs parameter investigation)")
    
    return strategies

def test_relaxed_strategies():
    """Test all relaxed strategies with a realistic backtest."""
    
    print(f"\n\n🧪 TESTING RELAXED STRATEGIES")
    print("="*50)
    
    strategies = create_relaxed_strategies()
    
    print(f"Testing {len(strategies)} relaxed strategies...")
    print(f"{'Strategy':<25} {'Trades':<8} {'Return':<10} {'Status':<15}")
    print("-" * 70)
    
    results = {}
    working_count = 0
    
    for strategy_name, strategy in strategies.items():
        try:
            orchestrator = IntegratedOrchestrator()
            
            # Test with 1 month of data
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
                status = "✅ WORKING"
                working_count += 1
            else:
                status = "❌ Still 0 trades"
            
            print(f"{strategy_name:<25} {trades:<8} {return_pct:+7.2f}% {status:<15}")
            
            results[strategy_name] = {
                'trades': trades,
                'return': return_pct,
                'status': 'working' if trades > 0 else 'failed'
            }
            
        except Exception as e:
            error_msg = str(e)[:20] + "..." if len(str(e)) > 20 else str(e)
            print(f"{strategy_name:<25} {'ERROR':<8} {'':>8} ❌ {error_msg}")
            results[strategy_name] = {'trades': 0, 'return': 0.0, 'status': 'error', 'error': str(e)}
    
    # Summary
    print(f"\n📊 RELAXED PARAMETER RESULTS:")
    print(f"  Working strategies: {working_count}/{len(strategies)}")
    print(f"  Success rate: {working_count/len(strategies)*100:.0f}%")
    
    if working_count >= 3:
        print(f"  🎉 EXCELLENT! Most strategies now work with relaxed parameters")
        print(f"  📋 Action: Apply these parameter sets to fix the foundation")
    elif working_count >= 2:
        print(f"  ✅ GOOD! Significant improvement with relaxed parameters")
        print(f"  📋 Action: Use working strategies, investigate remaining issues")
    elif working_count >= 1:
        print(f"  ⚠️ PARTIAL: Some improvement, but more work needed")
        print(f"  📋 Action: Focus on working strategies, may need strategy redesign")
    else:
        print(f"  ❌ POOR: Parameters not the only issue")
        print(f"  📋 Action: Deeper investigation needed")
    
    return results

def test_longer_periods():
    """Test if relaxed strategies work better with longer data periods."""
    
    print(f"\n\n📅 TESTING LONGER PERIODS WITH RELAXED PARAMETERS")
    print("="*65)
    
    strategies = create_relaxed_strategies()
    
    # Test 3 months vs 6 months
    test_periods = [
        ("1 month", datetime(2024, 1, 1), datetime(2024, 1, 31)),
        ("3 months", datetime(2024, 1, 1), datetime(2024, 3, 31)),
        ("6 months", datetime(2024, 1, 1), datetime(2024, 6, 30)),
    ]
    
    print(f"{'Strategy':<25} {'1M':<6} {'3M':<6} {'6M':<6} {'Best Period':<15}")
    print("-" * 70)
    
    all_results = {}
    
    for strategy_name, strategy in strategies.items():
        period_results = {}
        
        for period_name, start_date, end_date in test_periods:
            try:
                orchestrator = IntegratedOrchestrator()
                
                backtest_results = orchestrator.run_backtest(
                    symbols=["AAPL"],
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    initial_capital=1000
                )
                
                trades = backtest_results.total_trades
                period_results[period_name] = trades
                
            except Exception:
                period_results[period_name] = 0
        
        # Format results
        result_1m = f"{period_results.get('1 month', 0)}t"
        result_3m = f"{period_results.get('3 months', 0)}t" 
        result_6m = f"{period_results.get('6 months', 0)}t"
        
        # Find best period
        max_trades = max(period_results.values())
        best_periods = [p for p, t in period_results.items() if t == max_trades and t > 0]
        best_period = best_periods[0] if best_periods else "None work"
        
        print(f"{strategy_name:<25} {result_1m:<6} {result_3m:<6} {result_6m:<6} {best_period:<15}")
        all_results[strategy_name] = period_results
    
    return all_results

if __name__ == "__main__":
    print("🔧 SYSTEMATIC STRATEGY PARAMETER FIX")
    print("="*70)
    print("Goal: Fix the 4 failing strategies with realistic parameters\n")
    
    # Test relaxed parameters
    relaxed_results = test_relaxed_strategies()
    
    # Test longer periods  
    period_results = test_longer_periods()
    
    print(f"\n\n🎯 FOUNDATION FIX SUMMARY:")
    print("="*50)
    print("The 4 failing strategies had overly conservative parameters.")
    print("Relaxed parameters should restore basic signal generation.")
    print("Next step: Apply successful parameter combinations to fix foundation.")