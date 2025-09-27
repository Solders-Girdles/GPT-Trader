#!/usr/bin/env python3
"""
Debug each optimization component to find and fix integration issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy, MeanReversionParams
from bot.strategy.signal_filters import SignalQualityFilter, create_adaptive_filter

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def debug_signal_filters():
    """Debug why signal filters break execution."""
    
    print("\n" + "="*80)
    print("DEBUGGING SIGNAL FILTERS")
    print("="*80)
    
    symbol = "AAPL"
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 3, 31)
    
    # Test 1: Run without filters to establish baseline
    print("\n1. BASELINE TEST (No Filters)")
    print("-"*40)
    
    config_no_filter = UnifiedOptimizationConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=10000,
        quiet_mode=True,
        auto_apply_optimal_params=True,
        apply_signal_filters=False,  # NO FILTERS
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False
    )
    
    optimizer_no_filter = UnifiedOptimizer(config_no_filter)
    strategy = MeanReversionStrategy()
    
    # Hook into the optimizer to track signals
    original_process = optimizer_no_filter._run_daily_trading_loop
    signals_generated = []
    allocations_made = []
    
    def track_signals(bridge, market_data, trading_dates, results):
        # Track what happens
        print(f"Trading dates: {len(trading_dates)}")
        return original_process(bridge, market_data, trading_dates, results)
    
    optimizer_no_filter._run_daily_trading_loop = track_signals
    
    result_no_filter = optimizer_no_filter.run_backtest(strategy, [symbol])
    
    print(f"Return: {result_no_filter['metrics']['total_return']:.2f}%")
    print(f"Trades: {result_no_filter['metrics']['total_trades']}")
    print(f"Signals Generated: {result_no_filter['optimization']['signals_generated']}")
    
    # Test 2: Run with filters and debug
    print("\n2. WITH FILTERS TEST")
    print("-"*40)
    
    config_with_filter = UnifiedOptimizationConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=10000,
        quiet_mode=False,  # Show logs
        auto_apply_optimal_params=True,
        apply_signal_filters=True,  # ENABLE FILTERS
        filter_quality_threshold=0.3,  # Lower threshold for testing
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False,
        log_optimization_actions=True
    )
    
    optimizer_with_filter = UnifiedOptimizer(config_with_filter)
    
    # Track filter behavior
    print("\nRunning with filters enabled...")
    result_with_filter = optimizer_with_filter.run_backtest(strategy, [symbol])
    
    print(f"\nReturn: {result_with_filter['metrics']['total_return']:.2f}%")
    print(f"Trades: {result_with_filter['metrics']['total_trades']}")
    print(f"Signals Generated: {result_with_filter['optimization']['signals_generated']}")
    print(f"Signals Filtered: {result_with_filter['optimization']['signals_filtered']}")
    
    # Test 3: Test the filter directly
    print("\n3. DIRECT FILTER TEST")
    print("-"*40)
    
    # Create a simple test case
    import yfinance as yf
    data = yf.download(symbol, start="2024-01-01", end="2024-01-31", progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]
    
    # Add fake ATR
    data['atr'] = (data['high'] - data['low']).rolling(14).mean()
    
    # Create some test signals
    test_signals = pd.Series([0]*len(data), index=data.index)
    test_signals.iloc[10] = 1  # Buy signal
    test_signals.iloc[20] = -1  # Sell signal
    
    print(f"Original signals: {(test_signals != 0).sum()}")
    
    # Test the filter
    filter_config = create_adaptive_filter(10000)
    filtered = filter_config.filter_signals(data, test_signals, symbol)
    
    print(f"Filtered signals: {(filtered != 0).sum()}")
    
    # Check filter config
    print(f"\nFilter Configuration for $10,000 portfolio:")
    print(f"  Min volume ratio: {filter_config.config.min_volume_ratio}")
    print(f"  Max signals/week: {filter_config.config.max_signals_per_week}")
    print(f"  Use trend filter: {filter_config.config.use_trend_filter}")
    
    # Test 4: Debug the integration point
    print("\n4. INTEGRATION POINT DEBUG")
    print("-"*40)
    
    # Check what happens in _apply_quality_filters
    test_allocations = {symbol: 100}  # Want to buy 100 shares
    
    print(f"Input allocations: {test_allocations}")
    
    # Simulate what the unified optimizer does
    if optimizer_with_filter.signal_filter:
        # This is what happens in the code
        filtered_allocations = {}
        
        for sym, shares in test_allocations.items():
            # The filter expects market data for the symbol
            # If this fails, allocations become empty
            print(f"  Processing {sym}: {shares} shares")
            
            # This is likely where it breaks - 
            # the filter might be too strict or have a bug
            
    print("\nLikely issues:")
    print("1. Filter is removing ALL signals as low quality")
    print("2. Filter expects different data format")
    print("3. Filter thresholds are too strict")


def debug_regime_detection():
    """Debug why regime detection breaks execution."""
    
    print("\n" + "="*80)
    print("DEBUGGING REGIME DETECTION")
    print("="*80)
    
    symbol = "AAPL"
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 3, 31)
    
    # First verify regime detector exists and works
    from bot.strategy.regime_detector import MarketRegimeDetector
    
    print("1. Testing Regime Detector Directly")
    print("-"*40)
    
    import yfinance as yf
    data = yf.download("SPY", period="6mo", progress=False)
    
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(data)
    
    print(f"Current regime detected: {regime}")
    
    # Test with optimization
    print("\n2. Testing Regime in Optimizer")
    print("-"*40)
    
    config = UnifiedOptimizationConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=True,
        apply_signal_filters=False,  # Disable filters
        use_regime_detection=True,  # Enable regime
        regime_lookback=30,  # Shorter lookback
        use_trailing_stops=False,
        use_realistic_costs=False,
        log_optimization_actions=True
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    result = optimizer.run_backtest(strategy, [symbol])
    
    print(f"Return: {result['metrics']['total_return']:.2f}%")
    print(f"Regime changes: {result['optimization']['regime_changes']}")
    
    # Check regime history
    if result['optimization']['regime_history']:
        print(f"Regime history entries: {len(result['optimization']['regime_history'])}")
        print(f"First regime: {result['optimization']['regime_history'][0]}")


def debug_trailing_stops():
    """Debug why trailing stops don't work."""
    
    print("\n" + "="*80)
    print("DEBUGGING TRAILING STOPS")
    print("="*80)
    
    symbol = "AAPL"
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 3, 31)
    
    config = UnifiedOptimizationConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=True,
        apply_signal_filters=False,  # Keep disabled
        use_regime_detection=False,  # Keep disabled
        use_trailing_stops=True,  # Enable stops
        trailing_stop_atr_multiplier=2.0,
        use_profit_targets=True,
        profit_target_atr_multiplier=3.0,
        use_realistic_costs=False,
        log_optimization_actions=True
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    result = optimizer.run_backtest(strategy, [symbol])
    
    print(f"Return: {result['metrics']['total_return']:.2f}%")
    print(f"Trailing stops hit: {result['optimization']['trailing_stops_hit']}")
    print(f"Profit targets hit: {result['optimization']['profit_targets_hit']}")
    
    # Check if positions are being tracked
    print(f"\nPosition tracking: {len(optimizer.position_tracking)} positions")
    if optimizer.position_tracking:
        for sym, tracking in optimizer.position_tracking.items():
            print(f"  {sym}: Entry=${tracking['entry_price']:.2f}, "
                  f"Stop=${tracking['stop_loss']:.2f}, "
                  f"Target=${tracking['take_profit']:.2f}")


def debug_transaction_costs():
    """Debug realistic transaction costs."""
    
    print("\n" + "="*80)
    print("DEBUGGING TRANSACTION COSTS")
    print("="*80)
    
    symbol = "AAPL"
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 3, 31)
    
    # Test with different cost levels
    for spread_bps in [0, 5, 10, 20]:
        print(f"\nTesting with {spread_bps} bps spread:")
        
        config = UnifiedOptimizationConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=10000,
            quiet_mode=True,
            auto_apply_optimal_params=True,
            apply_signal_filters=False,
            use_regime_detection=False,
            use_trailing_stops=False,
            use_realistic_costs=True,
            spread_bps=spread_bps,
            slippage_bps=3.0,
            market_impact_bps=2.0
        )
        
        optimizer = UnifiedOptimizer(config)
        strategy = MeanReversionStrategy()
        
        result = optimizer.run_backtest(strategy, [symbol])
        
        total_costs = result['optimization']['total_transaction_costs']
        returns = result['metrics']['total_return']
        
        print(f"  Return: {returns:.2f}%")
        print(f"  Total costs: ${total_costs:.2f}")
        print(f"  Cost impact: {total_costs/10000*100:.2f}% of capital")


def fix_signal_filter_integration():
    """Attempt to fix the signal filter integration."""
    
    print("\n" + "="*80)
    print("ATTEMPTING TO FIX SIGNAL FILTERS")
    print("="*80)
    
    # The issue is likely in _apply_quality_filters method
    # Let's check what's happening
    
    print("Checking _apply_quality_filters implementation...")
    
    # Read the method to understand the issue
    from bot.integration.unified_optimizer import UnifiedOptimizer
    import inspect
    
    source = inspect.getsource(UnifiedOptimizer._apply_quality_filters)
    print("\nCurrent implementation:")
    print(source[:500] + "...")
    
    print("\nLikely fixes needed:")
    print("1. Ensure signal Series has correct index matching market data")
    print("2. Check that filter_signals returns non-empty result")
    print("3. Verify quality threshold isn't too high")
    print("4. Make sure ATR exists in market data for filter")


def main():
    """Run all debug tests."""
    
    print("="*80)
    print("OPTIMIZATION COMPONENT DEBUGGING")
    print("="*80)
    print(f"Debug Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Debug each component
    debug_signal_filters()
    debug_regime_detection()
    debug_trailing_stops()
    debug_transaction_costs()
    fix_signal_filter_integration()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()