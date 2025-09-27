#!/usr/bin/env python3
"""Fix failing strategies by testing longer periods and relaxed parameters - setup first."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.strategy.mean_reversion import MeanReversionStrategy  
from bot.strategy.momentum import MomentumStrategy
from bot.strategy.optimized_ma import OptimizedMAStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_longer_periods():
    """Test failing strategies with longer data periods - basic setup validation."""
    
    print("🔧 FOUNDATION SETUP: Testing Longer Periods")
    print("="*60)
    
    failing_strategies = [
        ("enhanced_trend_breakout", EnhancedTrendBreakoutStrategy()),
        ("mean_reversion", MeanReversionStrategy()),
        ("momentum", MomentumStrategy()),
        ("optimized_ma", OptimizedMAStrategy()),
    ]
    
    # Test different period lengths
    test_periods = [
        ("6 months", datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ("3 months", datetime(2024, 1, 1), datetime(2024, 3, 31)),
    ]
    
    print(f"{'Strategy':<20} {'3 Month':<10} {'6 Month':<10} {'Status':<15}")
    print("-" * 70)
    
    results = {}
    
    for strategy_name, strategy in failing_strategies:
        strategy_results = {}
        
        for period_name, start_date, end_date in test_periods:
            try:
                orchestrator = IntegratedOrchestrator()
                
                backtest_results = orchestrator.run_backtest(
                    symbols=["AAPL"],
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    initial_capital=1000  # Standard test amount
                )
                
                trades = backtest_results.total_trades
                strategy_results[period_name] = trades
                
            except Exception as e:
                strategy_results[period_name] = f"ERROR: {str(e)[:20]}"
        
        # Format results
        result_3m = strategy_results.get("3 months", 0)
        result_6m = strategy_results.get("6 months", 0) 
        
        result_3m_str = f"{result_3m}t" if isinstance(result_3m, int) else "ERR"
        result_6m_str = f"{result_6m}t" if isinstance(result_6m, int) else "ERR"
        
        # Overall status
        if isinstance(result_3m, int) and isinstance(result_6m, int):
            if result_3m > 0 or result_6m > 0:
                status = "✅ Fixable"
            else:
                status = "⚠️ No signals"
        else:
            status = "❌ Errors"
        
        print(f"{strategy_name:<20} {result_3m_str:<10} {result_6m_str:<10} {status:<15}")
        results[strategy_name] = strategy_results
    
    return results

def test_relaxed_parameters():
    """Test strategies with relaxed parameters - setup tuning."""
    
    print(f"\n\n🎛️ FOUNDATION SETUP: Testing Relaxed Parameters")
    print("="*60)
    
    # Create strategies with relaxed parameters
    relaxed_strategies = []
    
    # Enhanced Trend Breakout - try with shorter lookback
    try:
        # Check what parameters it accepts
        relaxed_strategies.append(("enhanced_trend_breakout_relaxed", EnhancedTrendBreakoutStrategy()))
    except Exception as e:
        print(f"  ⚠️ Enhanced trend breakout parameter adjustment failed: {e}")
    
    # Mean Reversion - try with more sensitive RSI levels
    try:
        from bot.strategy.mean_reversion import MeanReversionParams
        relaxed_mr_params = MeanReversionParams(
            rsi_period=10,  # Shorter period for faster signals
            oversold_threshold=40.0,  # Less extreme oversold (was 30)
            overbought_threshold=60.0  # Less extreme overbought (was 70)
        )
        relaxed_strategies.append(("mean_reversion_relaxed", MeanReversionStrategy(relaxed_mr_params)))
    except Exception as e:
        print(f"  ⚠️ Mean reversion parameter adjustment failed: {e}")
    
    # Momentum - try with lower momentum threshold
    try:
        from bot.strategy.momentum import MomentumParams
        relaxed_momentum_params = MomentumParams(
            roc_period=10,  # Shorter lookback
            momentum_threshold=1.5,  # Lower threshold (was 3.0)
            volume_threshold=1.2  # Lower volume requirement (was 1.5)
        )
        relaxed_strategies.append(("momentum_relaxed", MomentumStrategy(relaxed_momentum_params)))
    except Exception as e:
        print(f"  ⚠️ Momentum parameter adjustment failed: {e}")
    
    # Optimized MA - try with closer moving averages
    try:
        from bot.strategy.optimized_ma import OptimizedMAParams
        relaxed_ma_params = OptimizedMAParams(
            fast_period=5,   # Faster MA (was 10)
            slow_period=15   # Closer slow MA (was 30)
        )
        relaxed_strategies.append(("optimized_ma_relaxed", OptimizedMAStrategy(relaxed_ma_params)))
    except Exception as e:
        print(f"  ⚠️ Optimized MA parameter adjustment failed: {e}")
    
    if not relaxed_strategies:
        print("  ❌ No relaxed parameter strategies could be created")
        return {}
    
    print(f"\nTesting {len(relaxed_strategies)} strategies with relaxed parameters...")
    print(f"{'Strategy':<25} {'Trades':<8} {'Status':<15}")
    print("-" * 50)
    
    results = {}
    
    for strategy_name, strategy in relaxed_strategies:
        try:
            orchestrator = IntegratedOrchestrator()
            
            backtest_results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 31),  # 3 months
                strategy=strategy,
                initial_capital=1000
            )
            
            trades = backtest_results.total_trades
            status = "✅ Working" if trades > 0 else "⚠️ No signals"
            
            print(f"{strategy_name:<25} {trades:<8} {status:<15}")
            results[strategy_name] = trades
            
        except Exception as e:
            print(f"{strategy_name:<25} {'ERROR':<8} ❌ {str(e)[:20]}")
            results[strategy_name] = f"ERROR: {e}"
    
    return results

def comprehensive_foundation_setup():
    """Comprehensive foundation setup test for failing strategies."""
    
    print("🏗️ COMPREHENSIVE FOUNDATION SETUP TEST")
    print("="*70)
    print("Goal: Establish basic viability before optimization")
    
    # Test 1: Longer periods
    period_results = test_longer_periods()
    
    # Test 2: Relaxed parameters  
    relaxed_results = test_relaxed_parameters()
    
    # Analysis
    print(f"\n\n📊 FOUNDATION SETUP ANALYSIS")
    print("="*50)
    
    print(f"\n🔍 Period Length Impact:")
    working_with_longer = 0
    for strategy, results in period_results.items():
        result_3m = results.get("3 months", 0)
        result_6m = results.get("6 months", 0)
        if (isinstance(result_3m, int) and result_3m > 0) or (isinstance(result_6m, int) and result_6m > 0):
            working_with_longer += 1
            print(f"  ✅ {strategy}: Benefits from longer periods")
        else:
            print(f"  ❌ {strategy}: Still no signals with longer periods")
    
    print(f"\n🎛️ Parameter Relaxation Impact:")
    working_with_relaxed = 0
    for strategy, result in relaxed_results.items():
        if isinstance(result, int) and result > 0:
            working_with_relaxed += 1
            print(f"  ✅ {strategy}: Works with relaxed parameters")
        else:
            print(f"  ❌ {strategy}: Still failing with relaxed parameters")
    
    # Foundation assessment
    total_strategies = 4
    potential_fixes = working_with_longer + working_with_relaxed
    
    print(f"\n🏗️ FOUNDATION SETUP STATUS:")
    if potential_fixes >= 3:
        print(f"  ✅ EXCELLENT: {potential_fixes}/4 strategies can be fixed")
        print(f"  📋 Action: Apply successful parameter/period combinations")
    elif potential_fixes >= 2:
        print(f"  ⚠️ GOOD: {potential_fixes}/4 strategies can be fixed")  
        print(f"  📋 Action: Focus on working strategies, investigate others")
    elif potential_fixes >= 1:
        print(f"  ⚠️ LIMITED: {potential_fixes}/4 strategies can be fixed")
        print(f"  📋 Action: May need strategy redesign for others")
    else:
        print(f"  ❌ POOR: No strategies can be easily fixed")
        print(f"  📋 Action: Need deeper investigation or strategy replacement")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"  1. Apply successful parameter combinations")
    print(f"  2. Test working combinations across portfolio sizes")
    print(f"  3. Update strategy configurations")
    print(f"  4. Rerun foundation stress test")

if __name__ == "__main__":
    comprehensive_foundation_setup()