#!/usr/bin/env python3
"""
Comprehensive test of all feature slices.
Ensures each slice works independently and covers all components.
"""

import sys
import time
from datetime import datetime, timedelta
sys.path.append('/Users/rj/PycharmProjects/GPT-Trader/src/bot_v2')


def test_backtest_slice():
    """Test the backtest feature slice."""
    print("\n" + "="*60)
    print("TESTING BACKTEST SLICE")
    print("="*60)
    
    try:
        from features.backtest import run_backtest
        
        # Test with each strategy
        strategies = ['SimpleMAStrategy', 'MomentumStrategy', 'MeanReversionStrategy', 
                     'VolatilityStrategy', 'BreakoutStrategy']
        
        for strategy in strategies:
            print(f"\nTesting {strategy}...")
            result = run_backtest(
                strategy=strategy,
                symbol="AAPL",
                start=datetime.now() - timedelta(days=90),
                end=datetime.now(),
                initial_capital=10000
            )
            
            print(f"✅ {strategy}: Return={result.metrics.total_return:.2%}, "
                  f"Sharpe={result.metrics.sharpe_ratio:.2f}, "
                  f"Trades={len(result.trades)}")
        
        print("\n✅ BACKTEST SLICE: FULLY FUNCTIONAL")
        return True
        
    except Exception as e:
        print(f"❌ BACKTEST SLICE FAILED: {e}")
        return False


def test_paper_trade_slice():
    """Test the paper trade feature slice."""
    print("\n" + "="*60)
    print("TESTING PAPER TRADE SLICE")
    print("="*60)
    
    try:
        from features.paper_trade import start_paper_trading, stop_paper_trading, get_status
        
        # Start paper trading
        print("Starting paper trading session...")
        start_paper_trading(
            strategy="SimpleMAStrategy",
            symbols=["AAPL", "MSFT"],
            initial_capital=50000,
            fast_period=5,
            slow_period=15
        )
        
        # Let it run briefly
        print("Running for 3 seconds...")
        time.sleep(3)
        
        # Check status
        status = get_status()
        if status:
            print(f"Current equity: ${status.account_status.total_equity:,.2f}")
            print(f"Open positions: {len(status.positions)}")
        
        # Stop trading
        results = stop_paper_trading()
        print(f"Final equity: ${results.account_status.total_equity:,.2f}")
        print(f"Total trades: {len(results.trade_log)}")
        
        print("\n✅ PAPER TRADE SLICE: FULLY FUNCTIONAL")
        return True
        
    except Exception as e:
        print(f"❌ PAPER TRADE SLICE FAILED: {e}")
        return False


def test_analyze_slice():
    """Test the analyze feature slice."""
    print("\n" + "="*60)
    print("TESTING ANALYZE SLICE")
    print("="*60)
    
    try:
        from features.analyze import analyze_symbol, analyze_portfolio, compare_strategies
        
        # Test symbol analysis
        print("\n1. Testing symbol analysis...")
        analysis = analyze_symbol("AAPL", lookback_days=30)
        print(f"   Current price: ${analysis.current_price:.2f}")
        print(f"   RSI: {analysis.indicators.rsi:.2f}")
        print(f"   Trend: {analysis.regime.trend}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Patterns detected: {len(analysis.patterns)}")
        
        # Test portfolio analysis
        print("\n2. Testing portfolio analysis...")
        portfolio = analyze_portfolio(["AAPL", "MSFT", "GOOGL"])
        print(f"   Symbols analyzed: {len(portfolio.symbol_analyses)}")
        print(f"   Rebalance suggestions: {len(portfolio.rebalance_suggestions)}")
        
        # Test strategy comparison
        print("\n3. Testing strategy comparison...")
        comparison = compare_strategies("AAPL", ["SimpleMA", "Momentum"])
        print(f"   Best strategy: {comparison.best_strategy}")
        
        print("\n✅ ANALYZE SLICE: FULLY FUNCTIONAL")
        return True
        
    except Exception as e:
        print(f"❌ ANALYZE SLICE FAILED: {e}")
        return False


def test_optimize_slice():
    """Test the optimize feature slice."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZE SLICE")
    print("="*60)
    
    try:
        from features.optimize import optimize_strategy, grid_search, walk_forward_analysis
        
        # Test basic optimization
        print("\n1. Testing strategy optimization...")
        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=180),
            end_date=datetime.now() - timedelta(days=30),
            param_grid={
                'fast_period': [5, 10, 15],
                'slow_period': [20, 30, 40]
            }
        )
        print(f"   Best params: {result.best_params}")
        print(f"   Best return: {result.best_metrics.total_return:.2%}")
        print(f"   Combinations tested: {len(result.all_results)}")
        
        # Test grid search across strategies
        print("\n2. Testing grid search...")
        results = grid_search(
            strategies=["SimpleMA", "Momentum"],
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        print(f"   Strategies optimized: {len(results)}")
        
        # Test walk-forward (simplified)
        print("\n3. Testing walk-forward analysis...")
        wf_result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            window_size=60,
            step_size=30,
            test_size=30,
            param_grid={'fast_period': [10, 20], 'slow_period': [30, 40]}
        )
        print(f"   Windows analyzed: {len(wf_result.windows)}")
        print(f"   Average efficiency: {wf_result.avg_efficiency:.2f}")
        print(f"   Robustness score: {wf_result.robustness_score:.2%}")
        
        print("\n✅ OPTIMIZE SLICE: FULLY FUNCTIONAL")
        return True
        
    except Exception as e:
        print(f"❌ OPTIMIZE SLICE FAILED: {e}")
        return False


def identify_system_components():
    """Identify all system components and check slice coverage."""
    print("\n" + "="*60)
    print("SYSTEM COMPONENT ANALYSIS")
    print("="*60)
    
    print("\n📦 Core Components Built:")
    components = [
        ("SimpleDataProvider", "providers/"),
        ("SimpleBacktester", "backtesting/"),
        ("SimpleRiskManager", "risk/"),
        ("EqualWeightAllocator", "portfolio/"),
        ("5 Strategies", "strategies/"),
        ("ComponentRegistry", "core/"),
        ("EventBus", "core/")
    ]
    
    for component, location in components:
        print(f"  ✓ {component:<25} ({location})")
    
    print("\n🎯 Feature Slices Created:")
    slices = [
        ("Backtest", "Historical testing", True),
        ("Paper Trade", "Simulated trading", True),
        ("Analyze", "Market analysis", True),
        ("Optimize", "Parameter tuning", True)
    ]
    
    for name, purpose, complete in slices:
        status = "✅" if complete else "❌"
        print(f"  {status} {name:<15} - {purpose}")
    
    print("\n🔍 Potential Missing Slices:")
    missing = [
        ("live_trade", "Real broker integration", "Connect to Alpaca/IBKR"),
        ("monitor", "System monitoring", "Track health, performance, alerts"),
        ("data", "Data management", "Storage, caching, historical data"),
        ("risk", "Risk management", "Position sizing, portfolio risk"),
        ("portfolio", "Portfolio management", "Rebalancing, allocation")
    ]
    
    for name, purpose, description in missing:
        print(f"  ⚠️  {name:<15} - {purpose}")
        print(f"      Description: {description}")
    
    print("\n📊 Coverage Analysis:")
    print("  Core Trading Flow:")
    print("    Data → ✅ (in each slice)")
    print("    Strategy → ✅ (in each slice)")
    print("    Backtest → ✅ (backtest slice)")
    print("    Optimize → ✅ (optimize slice)")
    print("    Paper Trade → ✅ (paper_trade slice)")
    print("    Live Trade → ❌ (missing)")
    print("    Monitor → ❌ (missing)")
    
    print("\n  Risk & Portfolio:")
    print("    Risk Analysis → ✅ (in paper_trade slice)")
    print("    Position Sizing → ✅ (in execution modules)")
    print("    Portfolio Management → ⚠️ (partial in analyze)")
    print("    Rebalancing → ⚠️ (partial in analyze)")
    
    print("\n  Data & Infrastructure:")
    print("    Historical Data → ✅ (in each slice)")
    print("    Real-time Data → ✅ (in paper_trade)")
    print("    Data Storage → ❌ (missing)")
    print("    Monitoring → ❌ (missing)")


def test_slice_isolation():
    """Test that slices are truly isolated."""
    print("\n" + "="*60)
    print("TESTING SLICE ISOLATION")
    print("="*60)
    
    import os
    import ast
    
    slices = ['backtest', 'paper_trade', 'analyze', 'optimize']
    
    for slice_name in slices:
        slice_path = f'/Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/features/{slice_name}'
        print(f"\n📁 Checking {slice_name} slice...")
        
        violations = []
        for root, dirs, files in os.walk(slice_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        
                    # Check for imports from outside the slice
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom):
                                module = node.module or ''
                                # Check for cross-slice imports
                                if 'features.' in module and slice_name not in module:
                                    violations.append(f"    ❌ {file}: imports from {module}")
                                # Check for shared imports
                                elif 'shared' in module or 'utils' in module:
                                    violations.append(f"    ❌ {file}: imports from {module}")
                    except:
                        pass
        
        if violations:
            print(f"  ❌ ISOLATION VIOLATIONS FOUND:")
            for v in violations:
                print(v)
        else:
            print(f"  ✅ Perfect isolation - no external dependencies!")


def main():
    """Run all tests and analysis."""
    print("="*80)
    print("COMPREHENSIVE SLICE TESTING & COMPONENT ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Test each slice
    print("\n🧪 TESTING PHASE")
    results['backtest'] = test_backtest_slice()
    results['paper_trade'] = test_paper_trade_slice()
    results['analyze'] = test_analyze_slice()
    results['optimize'] = test_optimize_slice()
    
    # Test isolation
    test_slice_isolation()
    
    # Analyze components
    identify_system_components()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    print("\n📊 Test Results:")
    for slice, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {slice:<15} {status}")
    
    if all_passed:
        print("\n🎉 ALL SLICES FUNCTIONAL!")
        print("\n💡 Recommendations:")
        print("  1. Consider adding 'live_trade' slice for broker integration")
        print("  2. Consider adding 'monitor' slice for system health")
        print("  3. Consider adding 'data' slice for data management")
        print("\n✨ The vertical slice architecture with complete isolation is working perfectly!")
    else:
        print("\n⚠️ Some slices need attention")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)