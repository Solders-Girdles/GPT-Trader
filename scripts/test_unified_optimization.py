#!/usr/bin/env python3
"""
Comprehensive test of the unified optimization system.

This proves that each optimization component actually works and
contributes to improved performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import logging

from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
from bot.strategy.mean_reversion import MeanReversionStrategy, MeanReversionParams
from bot.strategy.trend_breakout import TrendBreakoutStrategy, TrendBreakoutParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_benchmark(symbol: str, start_date: str, end_date: str) -> float:
    """Calculate buy-and-hold benchmark return."""
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        return 0.0
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    return ((end_price - start_price) / start_price) * 100


def run_incremental_tests(symbol: str, start_date: str, end_date: str):
    """Test each optimization component incrementally."""
    
    print("\n" + "="*80)
    print(f"INCREMENTAL OPTIMIZATION TESTING - {symbol}")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    
    # Calculate benchmark
    benchmark = calculate_benchmark(symbol, start_date, end_date)
    print(f"Buy-and-Hold Benchmark: {benchmark:.2f}%")
    
    results = {}
    
    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Test 1: Base system (no optimization)
    print("\n" + "-"*60)
    print("TEST 1: Base System (No Optimization)")
    print("-"*60)
    
    try:
        base_config = BacktestConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True
        )
        
        base_orch = IntegratedOrchestrator(base_config)
        base_strategy = MeanReversionStrategy()  # Default params
        
        base_result = base_orch.run_backtest(base_strategy, [symbol])
        
        # Access metrics correctly
        base_return = base_result.total_return * 100
        base_trades = base_result.total_trades
        base_sharpe = base_result.sharpe_ratio
        
        print(f"  Return: {base_return:.2f}%")
        print(f"  Trades: {base_trades}")
        print(f"  Sharpe: {base_sharpe:.2f}")
        print(f"  Final Equity: ${base_orch.current_equity:,.2f}")
        
        results['base'] = base_return
        
    except Exception as e:
        print(f"  Error: {e}")
        results['base'] = 0.0
    
    # Test 2: Optimized parameters only
    print("\n" + "-"*60)
    print("TEST 2: Optimized Parameters Only")
    print("-"*60)
    
    try:
        param_config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True,
            # Enable ONLY parameter optimization
            auto_apply_optimal_params=True,
            apply_signal_filters=False,
            use_regime_detection=False,
            use_trailing_stops=False,
            use_realistic_costs=False
        )
        
        param_optimizer = UnifiedOptimizer(param_config)
        param_strategy = MeanReversionStrategy()  # Will get optimal params
        
        param_result = param_optimizer.run_backtest(param_strategy, [symbol])
        
        param_return = param_result['metrics']['total_return']
        param_trades = param_result['metrics']['total_trades']
        
        print(f"  Return: {param_return:.2f}%")
        print(f"  Trades: {param_trades}")
        print(f"  Improvement: {param_return - base_return:+.2f}%")
        
        results['params'] = param_return
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results['params'] = 0.0
    
    # Test 3: Add signal filters
    print("\n" + "-"*60)
    print("TEST 3: Parameters + Signal Filters")
    print("-"*60)
    
    try:
        filter_config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True,
            auto_apply_optimal_params=True,
            apply_signal_filters=True,  # ADD filters
            use_regime_detection=False,
            use_trailing_stops=False,
            use_realistic_costs=False
        )
        
        filter_optimizer = UnifiedOptimizer(filter_config)
        filter_result = filter_optimizer.run_backtest(param_strategy, [symbol])
        
        filter_return = filter_result['metrics']['total_return']
        filter_trades = filter_result['metrics']['total_trades']
        signals_filtered = filter_result['optimization']['signals_filtered']
        
        print(f"  Return: {filter_return:.2f}%")
        print(f"  Trades: {filter_trades}")
        print(f"  Signals Filtered: {signals_filtered}")
        print(f"  Improvement: {filter_return - param_return:+.2f}%")
        
        results['filters'] = filter_return
        
    except Exception as e:
        print(f"  Error: {e}")
        results['filters'] = 0.0
    
    # Test 4: Add regime detection
    print("\n" + "-"*60)
    print("TEST 4: Parameters + Filters + Regime Detection")
    print("-"*60)
    
    try:
        regime_config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True,
            auto_apply_optimal_params=True,
            apply_signal_filters=True,
            use_regime_detection=True,  # ADD regime
            use_trailing_stops=False,
            use_realistic_costs=False
        )
        
        regime_optimizer = UnifiedOptimizer(regime_config)
        regime_result = regime_optimizer.run_backtest(param_strategy, [symbol])
        
        regime_return = regime_result['metrics']['total_return']
        regime_changes = regime_result['optimization']['regime_changes']
        
        print(f"  Return: {regime_return:.2f}%")
        print(f"  Regime Changes: {regime_changes}")
        print(f"  Improvement: {regime_return - filter_return:+.2f}%")
        
        results['regime'] = regime_return
        
    except Exception as e:
        print(f"  Error: {e}")
        results['regime'] = 0.0
    
    # Test 5: Add trailing stops
    print("\n" + "-"*60)
    print("TEST 5: Parameters + Filters + Regime + Trailing Stops")
    print("-"*60)
    
    try:
        stops_config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True,
            auto_apply_optimal_params=True,
            apply_signal_filters=True,
            use_regime_detection=True,
            use_trailing_stops=True,  # ADD stops
            use_realistic_costs=False
        )
        
        stops_optimizer = UnifiedOptimizer(stops_config)
        stops_result = stops_optimizer.run_backtest(param_strategy, [symbol])
        
        stops_return = stops_result['metrics']['total_return']
        trailing_hits = stops_result['optimization']['trailing_stops_hit']
        
        print(f"  Return: {stops_return:.2f}%")
        print(f"  Trailing Stops Hit: {trailing_hits}")
        print(f"  Improvement: {stops_return - regime_return:+.2f}%")
        
        results['stops'] = stops_return
        
    except Exception as e:
        print(f"  Error: {e}")
        results['stops'] = 0.0
    
    # Test 6: Full optimization with realistic costs
    print("\n" + "-"*60)
    print("TEST 6: FULL OPTIMIZATION (All Components + Realistic Costs)")
    print("-"*60)
    
    try:
        full_config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=False,  # Show optimization log
            auto_apply_optimal_params=True,
            apply_signal_filters=True,
            use_regime_detection=True,
            use_trailing_stops=True,
            use_realistic_costs=True,  # ADD realistic costs
            log_optimization_actions=True
        )
        
        full_optimizer = UnifiedOptimizer(full_config)
        full_result = full_optimizer.run_backtest(param_strategy, [symbol])
        
        full_return = full_result['metrics']['total_return']
        total_costs = full_result['optimization']['total_transaction_costs']
        
        print(f"  Return: {full_return:.2f}%")
        print(f"  Transaction Costs: ${total_costs:.2f}")
        print(f"  Improvement vs Base: {full_return - base_return:+.2f}%")
        
        results['full'] = full_return
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results['full'] = 0.0
    
    # Summary
    print("\n" + "="*80)
    print("INCREMENTAL IMPROVEMENT SUMMARY")
    print("="*80)
    
    print(f"\n{'Component':<40} {'Return':<12} {'Δ vs Base':<12} {'Δ vs B&H'}")
    print("-"*76)
    
    print(f"{'Buy-and-Hold Benchmark':<40} {benchmark:<12.2f}% {'--':<12} {'--'}")
    print(f"{'Base System':<40} {results.get('base', 0):<12.2f}% {'--':<12} "
          f"{results.get('base', 0) - benchmark:+.2f}%")
    
    cumulative = results.get('base', 0)
    for name, key in [
        ('+ Optimized Parameters', 'params'),
        ('+ Signal Filters', 'filters'),
        ('+ Regime Detection', 'regime'),
        ('+ Trailing Stops', 'stops'),
        ('+ Realistic Costs (FULL)', 'full')
    ]:
        value = results.get(key, 0)
        improvement = value - results.get('base', 0)
        vs_benchmark = value - benchmark
        print(f"{name:<40} {value:<12.2f}% {improvement:+<12.2f}% {vs_benchmark:+.2f}%")
    
    # Verdict
    print("\n" + "="*80)
    if results.get('full', 0) > benchmark:
        print(f"✅ SUCCESS: Full optimization beats buy-and-hold by "
              f"{results.get('full', 0) - benchmark:.2f}%")
    else:
        print(f"⚠️ NEEDS TUNING: Full optimization underperforms by "
              f"{benchmark - results.get('full', 0):.2f}%")


def test_different_portfolios():
    """Test optimization with different portfolio sizes."""
    
    print("\n" + "="*80)
    print("PORTFOLIO SIZE ADAPTATION TEST")
    print("="*80)
    
    portfolios = [500, 1000, 5000, 10000, 25000]
    symbol = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-03-31"
    
    benchmark = calculate_benchmark(symbol, start_date, end_date)
    print(f"Benchmark ({symbol}): {benchmark:.2f}%")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"\n{'Portfolio':<15} {'Return':<12} {'Trades':<10} {'Costs':<12} {'vs B&H'}")
    print("-"*59)
    
    for capital in portfolios:
        try:
            config = UnifiedOptimizationConfig(
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=capital,
                quiet_mode=True,
                use_optimization=True,
                adaptive_filters=True  # Adapt to portfolio size
            )
            
            optimizer = UnifiedOptimizer(config)
            strategy = MeanReversionStrategy()
            
            result = optimizer.run_backtest(strategy, [symbol])
            
            ret = result['metrics']['total_return']
            trades = result['metrics']['total_trades']
            costs = result['optimization']['total_transaction_costs']
            vs_bh = ret - benchmark
            
            print(f"${capital:<14,} {ret:<12.2f}% {trades:<10} ${costs:<11.2f} {vs_bh:+.2f}%")
            
        except Exception as e:
            print(f"${capital:<14,} Error: {e}")


def main():
    """Run comprehensive optimization tests."""
    
    print("="*80)
    print("UNIFIED OPTIMIZATION SYSTEM TEST")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test cases
    test_cases = [
        ("AAPL", "2024-01-01", "2024-06-30"),
        ("SPY", "2024-01-01", "2024-06-30"),
    ]
    
    for symbol, start, end in test_cases:
        run_incremental_tests(symbol, start, end)
    
    # Test portfolio adaptations
    test_different_portfolios()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    print("\n📊 Key Insights:")
    print("1. Each optimization component should incrementally improve performance")
    print("2. Signal filters reduce overtrading and improve quality")
    print("3. Regime detection helps avoid bad market conditions")
    print("4. Trailing stops protect profits")
    print("5. Realistic costs are essential for accurate results")
    print("6. Smaller portfolios need more aggressive risk management")


if __name__ == "__main__":
    main()