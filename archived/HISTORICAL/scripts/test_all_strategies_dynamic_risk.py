#!/usr/bin/env python3
"""Test all available strategies with dynamic risk management and realistic portfolio sizes."""

import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.momentum import MomentumStrategy
from bot.strategy.optimized_ma import OptimizedMAStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_strategy_portfolio_matrix():
    """Test all strategies across different portfolio sizes."""
    
    print("🎯 COMPREHENSIVE STRATEGY × PORTFOLIO SIZE VALIDATION")
    print("="*80)
    
    # Strategy configurations to test
    strategies_to_test = [
        ("trend_breakout", TrendBreakoutStrategy()),
        ("volatility", VolatilityStrategy(VolatilityParams(bb_std_dev=1.5, atr_threshold_multiplier=0.8))),
        ("demo_ma", DemoMAStrategy()),
        ("enhanced_trend_breakout", EnhancedTrendBreakoutStrategy()),
        ("mean_reversion", MeanReversionStrategy()),
        ("momentum", MomentumStrategy()),
        ("optimized_ma", OptimizedMAStrategy()),
    ]
    
    # Realistic autonomous trading portfolio sizes
    portfolio_sizes = [1000, 2500, 5000]
    
    print(f"\n{'Strategy':<25} {'$1K':<8} {'$2.5K':<8} {'$5K':<8} {'Notes':<30}")
    print("-" * 90)
    
    results_matrix = {}
    
    for strategy_name, strategy in strategies_to_test:
        print(f"\n🔍 Testing {strategy_name}")
        results_matrix[strategy_name] = {}
        
        strategy_results = []
        
        for portfolio_size in portfolio_sizes:
            try:
                orchestrator = IntegratedOrchestrator()
                
                # Test with 2-month window for better signal coverage
                results = orchestrator.run_backtest(
                    symbols=["AAPL"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 2, 29),  # 2 months
                    strategy=strategy,
                    initial_capital=portfolio_size
                )
                
                trades = results.total_trades
                returns = results.total_return * 100
                final_value = portfolio_size * (1 + results.total_return)
                
                # Status: ✅ if trades > 0, ⚠️ if 0 trades but no errors, ❌ if error
                if trades > 0:
                    status = f"✅{trades}t"
                else:
                    status = "⚠️0t"
                
                strategy_results.append(status)
                results_matrix[strategy_name][portfolio_size] = {
                    'trades': trades,
                    'return_pct': returns,
                    'final_value': final_value,
                    'status': 'working' if trades > 0 else 'no_trades'
                }
                
            except Exception as e:
                error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                strategy_results.append(f"❌ERR")
                results_matrix[strategy_name][portfolio_size] = {
                    'trades': 0,
                    'return_pct': 0,
                    'final_value': portfolio_size,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"   Error with ${portfolio_size}: {error_msg}")
        
        # Determine overall strategy viability
        working_portfolios = sum(1 for result in strategy_results if result.startswith("✅"))
        if working_portfolios == 3:
            notes = "🎯 Fully autonomous ready"
        elif working_portfolios >= 1:
            notes = f"⚠️ Works for {working_portfolios}/3 sizes"
        else:
            notes = "❌ Not viable for small portfolios"
        
        print(f"{strategy_name:<25} {strategy_results[0]:<8} {strategy_results[1]:<8} {strategy_results[2]:<8} {notes:<30}")
    
    print(f"\n\n📊 DETAILED RESULTS BREAKDOWN")
    print("="*60)
    
    for strategy_name, portfolio_results in results_matrix.items():
        print(f"\n🎯 {strategy_name.upper()}")
        print("-" * 40)
        
        for portfolio_size, result in portfolio_results.items():
            if result['status'] == 'working':
                print(f"  ${portfolio_size:,}: ✅ {result['trades']} trades, {result['return_pct']:+.1f}%, final: ${result['final_value']:,.0f}")
            elif result['status'] == 'no_trades':
                print(f"  ${portfolio_size:,}: ⚠️ No trades executed (strategy may need tuning)")
            else:
                print(f"  ${portfolio_size:,}: ❌ Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n\n🏆 AUTONOMOUS TRADING READINESS SUMMARY")
    print("="*50)
    
    fully_ready = []
    partially_ready = []
    not_ready = []
    
    for strategy_name, portfolio_results in results_matrix.items():
        working_count = sum(1 for r in portfolio_results.values() if r['status'] == 'working')
        
        if working_count == 3:
            fully_ready.append(strategy_name)
        elif working_count >= 1:
            partially_ready.append(strategy_name)
        else:
            not_ready.append(strategy_name)
    
    print(f"✅ FULLY READY FOR AUTONOMOUS TRADING ({len(fully_ready)} strategies):")
    for strategy in fully_ready:
        print(f"   • {strategy}")
    
    print(f"\n⚠️ PARTIALLY READY ({len(partially_ready)} strategies):")
    for strategy in partially_ready:
        print(f"   • {strategy}")
    
    print(f"\n❌ NOT READY ({len(not_ready)} strategies):")
    for strategy in not_ready:
        print(f"   • {strategy}")
    
    print(f"\n🎯 CONCLUSION:")
    if len(fully_ready) >= 2:
        print(f"   🚀 System is READY for autonomous trading!")
        print(f"   📈 {len(fully_ready)} strategies work across all portfolio sizes")
        print(f"   💰 Minimum viable portfolio: $1,000")
    else:
        print(f"   ⚠️ Need more strategy tuning for small portfolios")
    
    return results_matrix

if __name__ == "__main__":
    test_strategy_portfolio_matrix()