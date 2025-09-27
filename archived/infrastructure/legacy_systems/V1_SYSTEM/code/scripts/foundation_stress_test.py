#!/usr/bin/env python3
"""Comprehensive foundation stress test for all strategies."""

import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.integration.orchestrator import IntegratedOrchestrator

def import_all_strategies():
    """Test importing all available strategies to find broken imports."""
    
    print("🔬 STRATEGY IMPORT STRESS TEST")
    print("="*50)
    
    strategy_imports = [
        ("trend_breakout", "from bot.strategy.trend_breakout import TrendBreakoutStrategy"),
        ("volatility", "from bot.strategy.volatility import VolatilityStrategy, VolatilityParams"),
        ("demo_ma", "from bot.strategy.demo_ma import DemoMAStrategy"),
        ("enhanced_trend_breakout", "from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy"),
        ("mean_reversion", "from bot.strategy.mean_reversion import MeanReversionStrategy"),
        ("momentum", "from bot.strategy.momentum import MomentumStrategy"),
        ("optimized_ma", "from bot.strategy.optimized_ma import OptimizedMAStrategy"),
        ("talib_optimized_ma", "from bot.strategy.talib_optimized_ma import TalibOptimizedMAStrategy"),
        ("ml_signal_strategy", "from bot.strategy.ml_signal_strategy import MLSignalStrategy"),
    ]
    
    imported_strategies = {}
    failed_imports = []
    
    for name, import_statement in strategy_imports:
        try:
            exec(import_statement, globals())
            print(f"  ✅ {name:<20} - Import successful")
            imported_strategies[name] = True
        except ImportError as e:
            print(f"  ❌ {name:<20} - Import failed: {str(e)}")
            failed_imports.append((name, str(e)))
        except Exception as e:
            print(f"  ⚠️  {name:<20} - Import error: {str(e)}")
            failed_imports.append((name, str(e)))
    
    print(f"\n📊 Import Results:")
    print(f"  ✅ Successful: {len(imported_strategies)}")
    print(f"  ❌ Failed: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n🔍 Failed Imports:")
        for name, error in failed_imports:
            print(f"    - {name}: {error}")
    
    return imported_strategies, failed_imports

def create_strategy_instances():
    """Create instances of all available strategies to test initialization."""
    
    print(f"\n\n🏗️  STRATEGY INITIALIZATION TEST")
    print("="*50)
    
    strategies = {}
    initialization_failures = []
    
    # Import successful strategies
    strategy_configs = [
        ("trend_breakout", "TrendBreakoutStrategy()"),
        ("volatility", "VolatilityStrategy(VolatilityParams(bb_std_dev=1.5, atr_threshold_multiplier=0.8))"),
        ("demo_ma", "DemoMAStrategy()"),
        ("enhanced_trend_breakout", "EnhancedTrendBreakoutStrategy()"),
        ("mean_reversion", "MeanReversionStrategy()"),
        ("momentum", "MomentumStrategy()"),
        ("optimized_ma", "OptimizedMAStrategy()"),
    ]
    
    for name, creation_code in strategy_configs:
        try:
            strategy = eval(creation_code)
            strategies[name] = strategy
            print(f"  ✅ {name:<20} - Initialization successful")
        except NameError:
            print(f"  ⏭️  {name:<20} - Skipped (import failed)")
        except Exception as e:
            print(f"  ❌ {name:<20} - Initialization failed: {str(e)}")
            initialization_failures.append((name, str(e)))
    
    print(f"\n📊 Initialization Results:")
    print(f"  ✅ Successful: {len(strategies)}")
    print(f"  ❌ Failed: {len(initialization_failures)}")
    
    return strategies, initialization_failures

def test_strategy_with_portfolios(strategy_name, strategy, portfolio_sizes=[500, 1000, 2500]):
    """Test a strategy across different portfolio sizes."""
    
    results = {}
    
    for portfolio_size in portfolio_sizes:
        try:
            orchestrator = IntegratedOrchestrator()
            
            # Use longer period for more realistic test
            backtest_results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),  # Full month
                strategy=strategy,
                initial_capital=portfolio_size
            )
            
            results[portfolio_size] = {
                'status': 'success',
                'trades': backtest_results.total_trades,
                'return': backtest_results.total_return * 100,
                'error': None
            }
            
        except Exception as e:
            results[portfolio_size] = {
                'status': 'error',
                'trades': 0,
                'return': 0.0,
                'error': str(e)
            }
    
    return results

def comprehensive_strategy_test():
    """Run comprehensive tests on all available strategies."""
    
    print(f"\n\n⚡ COMPREHENSIVE STRATEGY STRESS TEST")
    print("="*60)
    
    # Get working strategies
    strategies, init_failures = create_strategy_instances()
    
    if not strategies:
        print("❌ No strategies available for testing!")
        return
    
    print(f"\nTesting {len(strategies)} strategies with multiple portfolio sizes...")
    print(f"{'Strategy':<20} {'$500':<8} {'$1K':<8} {'$2.5K':<8} {'Status':<15}")
    print("-" * 70)
    
    all_results = {}
    overall_success_count = 0
    
    for strategy_name, strategy in strategies.items():
        try:
            results = test_strategy_with_portfolios(strategy_name, strategy)
            all_results[strategy_name] = results
            
            # Format results for display
            result_500 = results.get(500, {})
            result_1k = results.get(1000, {})
            result_2k5 = results.get(2500, {})
            
            status_500 = f"{result_500.get('trades', 0)}t" if result_500.get('status') == 'success' else "❌"
            status_1k = f"{result_1k.get('trades', 0)}t" if result_1k.get('status') == 'success' else "❌"
            status_2k5 = f"{result_2k5.get('trades', 0)}t" if result_2k5.get('status') == 'success' else "❌"
            
            # Overall status
            success_count = sum(1 for r in results.values() if r.get('status') == 'success' and r.get('trades', 0) > 0)
            if success_count == 3:
                overall_status = "✅ Perfect"
                overall_success_count += 1
            elif success_count >= 2:
                overall_status = "⚠️ Partial"
            elif success_count >= 1:
                overall_status = "⚠️ Limited"
            else:
                overall_status = "❌ Failed"
            
            print(f"{strategy_name:<20} {status_500:<8} {status_1k:<8} {status_2k5:<8} {overall_status:<15}")
            
        except Exception as e:
            print(f"{strategy_name:<20} {'ERR':<8} {'ERR':<8} {'ERR':<8} ❌ Error")
            all_results[strategy_name] = {'error': str(e)}
    
    # Summary analysis
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print(f"  Strategies tested: {len(strategies)}")
    print(f"  Perfect performers: {overall_success_count}")
    print(f"  Success rate: {overall_success_count/len(strategies)*100:.0f}%")
    
    # Detailed error analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    for strategy_name, results in all_results.items():
        if 'error' in results:
            print(f"\n❌ {strategy_name}: {results['error']}")
        else:
            print(f"\n✅ {strategy_name}:")
            for portfolio_size, result in results.items():
                if result['status'] == 'success':
                    trades = result['trades']
                    ret = result['return']
                    print(f"   ${portfolio_size:,}: {trades} trades, {ret:+.1f}% return")
                else:
                    print(f"   ${portfolio_size:,}: ❌ {result['error']}")
    
    # Foundation assessment
    if overall_success_count == len(strategies):
        print(f"\n🏆 FOUNDATION STATUS: EXCELLENT")
        print(f"   All strategies working across all portfolio sizes!")
    elif overall_success_count >= len(strategies) * 0.8:
        print(f"\n✅ FOUNDATION STATUS: GOOD")
        print(f"   Most strategies working, minor issues to address")
    elif overall_success_count >= len(strategies) * 0.5:
        print(f"\n⚠️ FOUNDATION STATUS: ACCEPTABLE")
        print(f"   Half of strategies working, significant issues remain")
    else:
        print(f"\n❌ FOUNDATION STATUS: NEEDS WORK")
        print(f"   Major strategy failures detected")
    
    return all_results

if __name__ == "__main__":
    # Run comprehensive strategy stress test
    imported_strategies, failed_imports = import_all_strategies()
    strategy_test_results = comprehensive_strategy_test()